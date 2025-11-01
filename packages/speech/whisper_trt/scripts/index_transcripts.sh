#!/bin/bash
# Index transcripts for Parenting AI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "📚 Indexing transcripts for Parenting AI..."
echo "=========================================="

python3 -m parenting_ai.transcript_indexer --verbose



