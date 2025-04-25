# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import argparse

from server.server import app, load_environments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the server.")
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    load_environments(args.config)

    # In production, use a WSGI server like Gunicorn or uWSGI instead of the development server.
    app.run(host='0.0.0.0', port=5050, debug=True)
