# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .arena_inprocess_env import IsaaclabArenaG1InProcessEnv

# The socket variant (Option B) is only importable when its stdlib-only client deps are
# present; guard it so the in-process default never depends on it.
try:
    from .arena_env import IsaaclabArenaG1Env
except Exception:  # noqa: BLE001
    IsaaclabArenaG1Env = None

# Maps an Arena env class name (used as ``env.*.init_params.id``) to its RLinf wrapper.
# Default is the in-process wrapper (RLinf in the same py3.12 venv as IsaacLab-Arena).
# All G1 locomanip tasks share one wrapper; they differ only in the Arena task/assets
# selected by ``init_params.id``.
REGISTER_ISAACLAB_ARENA_ENVS = {
    "LMDrillLiftRlD1": IsaaclabArenaG1InProcessEnv,
    "LMDrillLiftRl": IsaaclabArenaG1InProcessEnv,
    "LMBoxLiftRlD1": IsaaclabArenaG1InProcessEnv,
}

__all__ = [
    "REGISTER_ISAACLAB_ARENA_ENVS",
    "IsaaclabArenaG1InProcessEnv",
    "IsaaclabArenaG1Env",
]
