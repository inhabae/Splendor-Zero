from __future__ import annotations

from torch import Tensor, nn

from .state_schema import ACTION_DIM, STATE_DIM


class _MLPResBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.out_act(x + self.fc2(self.act(self.fc1(x))))


class MaskedPolicyValueNet(nn.Module):
    def __init__(
        self,
        input_dim: int = STATE_DIM,
        hidden_dim: int = 256,
        action_dim: int = ACTION_DIM,
        res_blocks: int = 0,
    ) -> None:
        super().__init__()
        if res_blocks < 0:
            raise ValueError(f"res_blocks must be >= 0, got {res_blocks}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.res_blocks = int(res_blocks)

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.residual_tower = nn.ModuleList([_MLPResBlock(hidden_dim=self.hidden_dim) for _ in range(self.res_blocks)])
        self.policy_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.value_head = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Tanh())
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head[0].bias)

    def export_model_kwargs(self) -> dict[str, int]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "res_blocks": self.res_blocks,
        }

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(x)
        for block in self.residual_tower:
            h = block(h)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return policy_logits, value
