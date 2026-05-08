#!/bin/bash
# Install SUMO + Python deps for Google Colab.
# Run in Colab: !bash scripts/colab_install_sumo.sh

set -e

# SUMO via apt (Colab is Ubuntu)
add-apt-repository ppa:sumo/stable -y 2>/dev/null || true
apt-get update -y
apt-get install -y sumo sumo-tools sumo-doc

# SUMO_HOME
export SUMO_HOME=/usr/share/sumo
echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc

# TraCI: traci is bundled with SUMO, add tools to path for other modules
export PYTHONPATH="${SUMO_HOME}/tools:${PYTHONPATH}"
echo "export PYTHONPATH=\"${SUMO_HOME}/tools:\${PYTHONPATH}\"" >> ~/.bashrc

echo "SUMO installed. SUMO_HOME=$SUMO_HOME"
