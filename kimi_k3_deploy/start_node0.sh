#!/usr/bin/env bash
# Rank 0 (10.170.168.1) - API server. Start this first, then start_node1.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_kimi_k3_tp8_pp2.sh" 0
