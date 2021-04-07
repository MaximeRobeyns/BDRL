#!/usr/bin/env bash
set -euo pipefail

docker run --rm -v $(pwd)/bdrl:/bdrl -v $(pwd)/source:/docs/source -p 8080:8080 docs-writer watch
