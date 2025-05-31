import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # <--- Importa optim per lo scheduler
from torch_geometric.nn import global_mean_pool, GINEConv
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score

class Graph_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout_rate=0.5, edge_channels=0):
        super(Graph_Net, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.edge_channels = edge_channels 

        # Layer lineare iniziale per mappare le feature di input alla dimensione nascosta
        self.lin_init = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            # Definisci l'MLP inline usando nn.Sequential
            mlp_for_gin = nn.Sequential(
                nn.Linear(hidden_channels, 2 * hidden_channels), 
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, hidden_channels)
            )
            self.convs.append(GINEConv(mlp_for_gin, edge_dim=edge_channels, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.lin_final = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if self.edge_channels > 0 and edge_attr is None:
            print(f"Warning: Model configured with edge_channels={self.edge_channels}, but edge_attr is None. Edge features will not be used.")

        x = self.lin_init(x)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin_final(x)
        return x

class GraphClassifier(pl.LightningModule):
    def __init__(self, num_node_features, num_classes, hidden_dim, num_layers, dropout_rate, learning_rate, edge_channels,
                 loss_type='cross_entropy', label_smoothing=0.0, gce_q=0.7): # Nuovi argomenti con default
        super().__init__()
        self.save_hyperparameters() # Questo salverà anche i nuovi argomenti
        self.graph_net = Graph_Net(num_node_features, hidden_dim, num_classes, num_layers, dropout_rate, edge_channels)
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.gce_q = gce_q

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        self._test_metrics_updated = False 

    def forward(self, x, edge_index, batch, edge_attr=None):
        return self.graph_net(x, edge_index, batch, edge_attr)

    def _label_smoothed_cross_entropy(self, pred, target, epsilon):
        # pred: log_softmax output
        # target: ground truth labels (integers)
        # epsilon: smoothing parameter

        n_classes = pred.size(1)
        smooth_target = torch.full_like(pred, epsilon / n_classes)
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - epsilon + (epsilon / n_classes))

        loss = F.kl_div(pred, smooth_target, reduction='batchmean')
        return loss

    def _generalized_cross_entropy(self, pred, target, q):
        # pred: output logits (NON log_softmax)
        # target: ground truth labels (integers)
        # q: parameter for GCE

        prob = F.softmax(pred, dim=1)
        prob_target = prob.gather(1, target.unsqueeze(1)).squeeze()

        # Clamp prob_target per stabilità numerica ed evitare 0^power issues
        prob_target = prob_target.clamp(min=1e-10) 

        loss = (1 - (prob_target ** q)) / q
        
        loss = torch.mean(loss)
        return loss

    def training_step(self, batch, batch_idx):
        x, edge_index, batch_map = batch.x, batch.edge_index, batch.batch
        edge_attr = getattr(batch, 'edge_attr', None)
        
        out = self.graph_net(x, edge_index, batch_map, edge_attr)
        labels = batch.y # Rinomina per chiarezza

        if self.loss_type == 'label_smoothing':
            preds = F.log_softmax(out, dim=1)
            loss = self._label_smoothed_cross_entropy(preds, labels, self.label_smoothing)
        elif self.loss_type == 'gce':
            loss = self._generalized_cross_entropy(out, labels, self.gce_q)
            preds = F.log_softmax(out, dim=1) # Usiamo preds log_softmax per le metriche
        else: # Default a cross_entropy standard
            preds = F.log_softmax(out, dim=1)
            loss = F.nll_loss(preds, labels)
        
        self.train_accuracy.update(preds, labels)
        self.train_f1.update(preds, labels)
        
        # Correggi il nome delle metriche per coerenza con lo scheduler
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        self.log('train_acc_step', self.train_accuracy, on_step=True, on_epoch=False, batch_size=batch.num_graphs)
        self.log('train_f1_step', self.train_f1, on_step=True, on_epoch=False, batch_size=batch.num_graphs)
        return loss

    def on_train_epoch_end(self):
        # Correggi il nome delle metriche per coerenza con lo scheduler
        self.log('train_acc_epoch', self.train_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1_epoch', self.train_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.train_accuracy.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, edge_index, batch_map = batch.x, batch.edge_index, batch.batch
        edge_attr = getattr(batch, 'edge_attr', None)
        
        out = self.graph_net(x, edge_index, batch_map, edge_attr)
        labels = batch.y # Rinomina per chiarezza

        if self.loss_type == 'label_smoothing':
            preds = F.log_softmax(out, dim=1)
            loss = self._label_smoothed_cross_entropy(preds, labels, self.label_smoothing)
        elif self.loss_type == 'gce':
            loss = self._generalized_cross_entropy(out, labels, self.gce_q)
            preds = F.log_softmax(out, dim=1) # Usiamo preds log_softmax per le metriche
        else: # Default a cross_entropy standard
            preds = F.log_softmax(out, dim=1)
            loss = F.nll_loss(preds, labels)
        
        self.val_accuracy.update(preds, labels)
        self.val_f1.update(preds, labels)
        
        # Correggi il nome delle metriche per coerenza con lo scheduler
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def on_validation_epoch_end(self):
        # Correggi il nome delle metriche per coerenza con lo scheduler
        self.log('val_acc_epoch', self.val_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_epoch', self.val_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.val_accuracy.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, edge_index, batch_map, labels = batch.x, batch.edge_index, batch.batch, batch.y 
        edge_attr = getattr(batch, 'edge_attr', None)

        out = self.graph_net(x, edge_index, batch_map, edge_attr)

        preds = F.log_softmax(out, dim=1)
        if labels is not None and labels.numel() > 0: 
            # Per consistenza con l'addestramento, usiamo la stessa loss_type
            if self.loss_type == 'label_smoothing':
                loss = self._label_smoothed_cross_entropy(preds, labels, self.label_smoothing)
            elif self.loss_type == 'gce':
                loss = self._generalized_cross_entropy(out, labels, self.gce_q) 
            else: # Default a cross_entropy standard
                loss = F.nll_loss(preds, labels)

            self.test_accuracy.update(preds, labels)
            self.test_f1.update(preds, labels)
            self._test_metrics_updated = True
            self.log('test_loss', loss, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
            return {"test_loss": loss}
        return {} 

    def on_test_epoch_end(self):
        if self._test_metrics_updated:
            self.log('final_test_acc', self.test_accuracy.compute(), on_step=False, on_epoch=True)
            self.log('final_test_f1', self.test_f1.compute(), on_step=False, on_epoch=True)
            self.test_accuracy.reset()
            self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Definisci lo scheduler per il learning rate
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',       # 'max' perché monitoriamo 'val_f1_epoch' (che vogliamo massimizzare)
                factor=0.5,       # Fattore di riduzione: il LR sarà moltiplicato per 0.5 (es. 0.001 -> 0.0005)
                patience=10,      # Numero di epoche senza miglioramento prima di ridurre il LR
                verbose=True,     # Stampa un messaggio quando il LR viene ridotto
                min_lr=1e-6       # Limite inferiore per il learning rate
            ),
            'monitor': 'val_f1_epoch', # La metrica da monitorare nel trainer. DEVE CORRISPONDERE ESATTAMENTE A QUELLA LOGGATA
            'interval': 'epoch',      # Lo scheduler controlla la metrica alla fine di ogni epoca
            'frequency': 1            # Controlla ogni epoca
        }
        
        # PyTorch Lightning si aspetta una lista di optimizer e una lista di scheduler
        return [optimizer], [scheduler]