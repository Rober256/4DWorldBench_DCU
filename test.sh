salloc -p client_dtk -J test -N 1 --exclusive \
  --ntasks-per-node=8 --cpus-per-task=16 \
  --gres=dcu:8 --time=10:00:00

module load compiler/dtk/26.04
export PATH="$HOME/.local/bin:$PATH"
