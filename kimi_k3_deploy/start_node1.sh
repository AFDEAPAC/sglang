#!/usr/bin/env bash
# Rank 1 (10.170.168.2) - second pipeline stage. Start right after start_node0.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_kimi_k3_tp8_pp2.sh" 1
