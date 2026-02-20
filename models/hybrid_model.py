import torch
import torch.nn as nn
import torch.nn.functional as F

from .classical_backbone import ClassicalBackbone
from .quantum_layer import QuantumLayer, n_qubits


class UQCCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Classical feature extractor
        self.backbone = ClassicalBackbone()

        # Classical → Quantum projection
        self.fc_quantum = nn.Linear(512, n_qubits)

        # Quantum layer
        self.quantum = QuantumLayer()

        # Fusion head
        self.fc_fusion = nn.Linear(512 + n_qubits, 256)
        self.bn_fusion = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.25)

        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):

        # Step 1: Extract classical features
        classical_features = self.backbone(x)   # (batch_size, 512)

        # Step 2: Prepare input for quantum layer
        quantum_input = torch.tanh(self.fc_quantum(classical_features))

        # Step 3: Quantum forward pass
        quantum_features = self.quantum(quantum_input).float()

        # Step 4: Fuse classical + quantum
        fused = torch.cat([classical_features, quantum_features], dim=1)

        fused = F.relu(self.bn_fusion(self.fc_fusion(fused)))
        fused = self.dropout(fused)

        # Step 5: Final classification
        output = self.fc_out(fused)

        return output