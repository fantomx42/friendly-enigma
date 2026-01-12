#!/bin/bash

# Check if 'ralph.ai' exists in /etc/hosts
if ! grep -q "ralph.ai" /etc/hosts; then
    echo "Adding ralph.ai to /etc/hosts..."
    echo "127.0.0.1 ralph.ai" >> /etc/hosts
fi

echo ""
echo "To apply the changes, run this script with sudo:"
echo "sudo ./setup_domain.sh"
