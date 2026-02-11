#!/bin/bash
if [ -f backend/Cargo.toml ]; then
  echo "Backend crate exists"
  exit 0
else
  echo "Backend crate missing"
  exit 1
fi
