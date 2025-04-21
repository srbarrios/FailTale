import argparse

from mcp_server.server import app, load_environments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP server.")
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    print("Starting MCP server...")
    load_environments(args.config)

    # In production, use a WSGI server like Gunicorn or uWSGI instead of the development server.
    app.run(host='0.0.0.0', port=5050, debug=True)
