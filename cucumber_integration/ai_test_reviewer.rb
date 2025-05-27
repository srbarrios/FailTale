# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

require 'faraday'
require 'json'
require 'yaml'
require 'uri'
require 'open3'

# AI Test Reviewer
#
# A client to interact with the FailTale Server API.
# Supports loading a list of target hosts from a YAML file and sending data
# to `/v1/collect` (for diagnostics gathering) and `/v1/analyze` (for root cause analysis).
#
# Usage assumes a YAML file containing the `hosts:` section with machine definitions
# and a running FailTale server.
class AITestReviewer
  attr_reader :base_url, :environment_file, :hosts, :timeout

  # Initializes the AITestReviewer.
  #
  # @param [String] base_url The base URL of the FailTale Server (e.g., http://localhost:5050).
  # @param [String] environment_file Path to a YAML file describing hosts and configuration.
  # @param [Integer] timeout Request timeout in seconds (default: 600).
  #
  # @raise [ArgumentError] If base_url or environment_file is missing or invalid.
  def initialize(base_url: 'http://localhost:5050', environment_file: '/root/spacewalk/testsuite/config/ai_environment.yml', timeout: 600)
    raise ArgumentError, 'Base URL for Server is required.' if base_url.nil? || base_url.strip.empty?
    raise ArgumentError, 'Environment configuration YAML file is required.' if environment_file.nil? || environment_file.strip.empty?
    raise ArgumentError, "YAML file not found at path: #{environment_file}" unless File.exist?(environment_file)

    @base_url = base_url.chomp('/')
    @collect_endpoint = "#{@base_url}/v1/collect"
    @analyze_endpoint = "#{@base_url}/v1/analyze"
    @timeout = timeout
    @environment_file = environment_file
    @hosts = load_hosts_from_yaml

    return if File.exist?('/tmp/ai_test_reviewer_setup')

    setup_ollama
    install_and_run_ai_test_reviewer
    File.write('/tmp/ai_test_reviewer_setup', '')
  end

  # Sends a request to the `/v1/collect` endpoint to retrieve diagnostic data from hosts.
  #
  # @param [String] test_report. Test report in Gherkin format.
  #
  # @return [Hash] Response from the FailTale endpoint.
  def collect(test_report)
    if @hosts.nil? || @hosts.empty?
      warn 'Cannot collect data: No hosts provided or loaded.'
      return { 'error' => 'No hosts provided or loaded.' }
    end

    payload = {
      'hosts' => @hosts,
      'test_report' => test_report
    }

    post(@collect_endpoint, payload)
  end

  # Sends a request to the `/v1/analyze` endpoint to retrieve root cause hints.
  #
  # @param [Hash] collected_data. Diagnostic data from hosts.
  # @param [String] screenshot. Base64-encoded screenshot of the test failure.
  # @param [String] test_report. Test report in Gherkin format.
  # @param [String] test_failure. Test failure returned by Cucumber.
  #
  # @return [Hash] Response from the FailTale endpoint.
  def analyze(collected_data, screenshot, test_report, test_failure)
    if collected_data.nil? || collected_data.empty?
      warn 'Analyze called with empty collected_data.'
      return { 'error' => 'collected_data is empty.' }
    end

    payload = {
      'collected_data' => collected_data,
      'screenshot' => screenshot,
      'test_report' => test_report,
      'test_failure' => test_failure
    }

    post(@analyze_endpoint, payload)
  end

  private

  # Loads the list of target hosts from the YAML environment file.
  #
  # The file must define a `hosts` key containing an array of hashes.
  # Each host may define:
  #   - hostname: [String] (required),
  #   - role: [String] (required),
  #   - ssh_username: [String] (optional),
  #   - ssh_password: [String] (optional),
  #   - mandatory: [Boolean] whether this host must be reachable.
  #
  # @return [Array<Hash>] Parsed host configurations or empty array if none defined.
  # @raise [ArgumentError] If the file is missing or unreadable.
  def load_hosts_from_yaml
    yaml_content = YAML.load_file(environment_file)
    yaml_content['hosts'] || []
  end

  # Sends a POST request with a JSON body to a given FailTale endpoint.
  #
  # @param [String] url The endpoint URL (`/v1/collect` or `/v1/analyze`).
  # @param [Hash] payload The data to send as JSON.
  #
  # @return [Hash] Response as a JSON if successful.
  def post(url, payload)
    connection = Faraday.new(url: url, request: { timeout: timeout }) do |faraday|
      faraday.request :json
      faraday.response :raise_error
      faraday.adapter Faraday.default_adapter
    end

    response = connection.post do |req|
      req.headers['Content-Type'] = 'application/json'
      req.body = payload
    end

    JSON.parse(response.body)
  rescue Faraday::Error => e
    error_message = "HTTP request to #{url} failed: #{e.message}"
    error_message += "\nResponse body: #{e.response[:body]}" if e.response&.[](:body)
    warn error_message
    { 'error' => error_message }
  rescue JSON::ParserError => e
    warn "Failed to parse JSON response from #{url}: #{e.message}"
    { 'error' => "Invalid JSON response from server: #{e.message}" }
  end

  # Installs Ollama and pull the required models.
  def setup_ollama
    hostname = URI.parse(@base_url).host
    raise "Invalid base URL: cannot extract hostname from #{@base_url}" unless hostname

    setup_script = <<~SCRIPT
      curl -fsSL https://ollama.com/install.sh | sh && \
      mkdir -p ~/.ollama && \
      echo "num-parallel: 6" > ~/.ollama/config.yaml && \
      sleep 5 && \
      ollama pull mistral
    SCRIPT

    stdout, stderr, status = Open3.capture3("ssh #{hostname.shellescape} #{setup_script.shellescape}")
    if status.success?
      puts "Ollama setup completed successfully on #{hostname}."
    else
      warn "Failed to set up Ollama on #{hostname}.\nSTDOUT: #{stdout}\nSTDERR: #{stderr}"
    end
  end

  # Installs FailTale container locally
  def install_and_run_ai_test_reviewer
    commands = [
      'zypper install -y docker',
      'systemctl enable docker',
      'systemctl start docker',
      'docker pull ghcr.io/srbarrios/failtale:latest',
      "docker run -d --rm --net=host -v /root/.ssh/id_ed25519:/root/.ssh/id_ed25519:ro -v #{@environment_file}:/app/config.yaml:ro --name failtale_service ghcr.io/srbarrios/failtale"
    ]

    commands.each do |cmd|
      stdout, stderr, status = Open3.capture3(cmd)
      warn "Command failed: #{cmd}\nSTDOUT: #{stdout}\nSTDERR: #{stderr}" unless status.success?
    end
  end
end
