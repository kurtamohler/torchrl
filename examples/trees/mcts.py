# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torchrl
import tensordict
import torch

# NOTE: I really should use MCTSForest instead of Tree, because it offers a
# friendlier API to interact with the tree.
def traverse_MCTS(forest, env, max_rollout_steps):
    """Performs one update step of an MCTS tree.

    Args:
        forest (torchrl.data.MCTSForest): Forest containing the tree
        env (torchrl.envs.EnvBase): Environment to performs actions in.
        max_rollout_steps (int): Maximum number of steps for each rollout.
    """
    current_node = tree
    done = False

    while not done:
        if current_node.is_leaf():
            if (current_node.visits > 0 and not current_node.is_terminal) or current_node.parent is None:
                actions = current_node.full_action_spec

env = torchrl.envs.ChessEnv(
    include_fen=True,
    include_hash=True,
    include_hash_inv=True,
)

forest = torchrl.data.MCTSForest()
forest.reward_keys = env.reward_keys
forest.done_keys = env.done_keys
forest.action_keys = env.action_keys

forest.observation_keys = ["fen_hash", "turn", "action_mask"]

td_reset = env.reset()

td0 = env.rollout(1)
td1 = env.rollout(1)

# Make sure the two rollouts are different
while (td0['next', 'fen_hash'] == td1['next', 'fen_hash']).all():
    td1 = env.rand_step(td_reset.clone())

forest.add(td0)
forest.add(td1)

tree = forest.get_tree(td_reset)
assert (tree.node_data['fen_hash'] == td_reset['fen_hash']).all()
assert (tree.subtree[0].node_data['fen_hash'] == td0['next', 'fen_hash'][-1]).all()
assert (tree.subtree[1].node_data['fen_hash'] == td1['next', 'fen_hash'][-1]).all()

td3_step = env.rand_step(td0[-1]['next']).exclude('reward').unsqueeze(0)
td3 = tensordict.LazyStackedTensorDict.cat([td0, td3_step])
forest.add(td3)

tree = forest.get_tree(td_reset)
assert (tree.node_data['fen_hash'] == td_reset['fen_hash']).all()
assert (tree.subtree[0].node_data['fen_hash'] == td0['next', 'fen_hash'][-1]).all()
assert (tree.subtree[1].node_data['fen_hash'] == td1['next', 'fen_hash'][-1]).all()
assert (tree.subtree[0].subtree[0].node_data['fen_hash'] == td3['next', 'fen_hash'][-1]).all()

#td4_step = env.rand_step(td3[-1]['next'])
#td4 = tensordict.LazyStackedTensorDict.cat([td3, td4_step.exclude('reward').unsqueeze(0)])
#
#forest.add(td4)
#
#tree = forest.get_tree(td_reset)
#assert (tree.node_data['fen_hash'] == td_reset['fen_hash']).all()
#assert (tree.subtree[0].node_data['fen_hash'] == td0['next', 'fen_hash'][-1]).all()
#assert (tree.subtree[1].node_data['fen_hash'] == td1['next', 'fen_hash'][-1]).all()
#assert (tree.subtree[0].subtree[0].node_data['fen_hash'] == td3['next', 'fen_hash'][-1]).all()
#assert (tree.subtree[0].subtree[0].subtree[0].node_data['fen_hash'] == td4['next', 'fen_hash'][-1]).all()