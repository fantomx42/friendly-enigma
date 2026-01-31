class RalphTerminalPage {
  constructor(page) {
    this.page = page;
    this.header = page.locator('h1');
    this.modeBtn = page.locator('#mode-btn');
    this.input = page.locator('#input');
    this.sendBtn = page.locator('#send-btn');
    this.log = page.locator('#log');
    this.activityIndicator = page.locator('#activity-indicator');
    this.activityText = page.locator('#activity-text');
    this.statusFooter = page.locator('#status');
    this.latencyFooter = page.locator('#latency');
    
    // Metrics
    this.metricTokens = page.locator('#metric-tokens');
    this.metricDuration = page.locator('#metric-duration');
    this.metricStatus = page.locator('#metric-status');
    
    // Sidebars
    this.planContainer = page.locator('#plan-container');
    this.codePreview = page.locator('#code-preview');
    
    // Quick Actions
    this.systemStatusChip = page.getByText('📊 System Status');
    this.explainTopicChip = page.getByText('🧠 Explain Topic');
    this.resetContextChip = page.getByText('🧹 Reset Context');
  }

  async goto() {
    await this.page.goto('/');
  }

  async sendCommand(command) {
    await this.input.fill(command);
    await this.sendBtn.click();
  }

  async sendCommandViaEnter(command) {
    await this.input.fill(command);
    await this.input.press('Enter');
  }

  async toggleMode() {
    await this.modeBtn.click();
  }

  async getLogText() {
    return await this.log.innerText();
  }

  async waitForStatus(status) {
    await this.page.waitForFunction(
      (s) => document.getElementById('status').innerText.includes(s),
      status
    );
  }

  async waitForActivity(text) {
    if (text) {
      await this.page.waitForFunction(
        (t) => document.getElementById('activity-text').innerText.includes(t),
        text
      );
    } else {
      await this.activityIndicator.waitFor({ state: 'visible' });
    }
  }

  async waitForActivityToFinish() {
    await this.activityIndicator.waitFor({ state: 'hidden', timeout: 60000 });
  }
}

module.exports = { RalphTerminalPage };
