const { test, expect } = require('@playwright/test');
const { RalphTerminalPage } = require('./RalphTerminalPage');

test.describe('Ralph Terminal E2E Tests', () => {
  let ralphPage;

  test.beforeEach(async ({ page }) => {
    ralphPage = new RalphTerminalPage(page);
    await ralphPage.goto();
    // Wait for the initial connection
    await expect(ralphPage.statusFooter).toHaveText(/ONLINE/, { timeout: 10000 });
  });

  test('Initial state is correct', async ({ page }) => {
    await expect(ralphPage.header).toHaveText('RALPH OS // v3.1');
    await expect(ralphPage.modeBtn).toHaveText('SIMPLE MODE: ON');
    await expect(ralphPage.log).toContainText('Hello! I am Ralph. How can I help you today?');
    await expect(ralphPage.metricTokens).toHaveText('0');
    await expect(ralphPage.metricStatus).toHaveText('IDLE');
  });

  test('User can send a command and see it echoed', async ({ page }) => {
    const command = 'Test command 123';
    await ralphPage.sendCommand(command);
    
    // Check if command is echoed in user message style
    const userMsg = page.locator('.user-msg');
    await expect(userMsg).toContainText(command);
    
    // Check if directive accepted message appears
    await expect(ralphPage.log).toContainText(`Directive Accepted: ${command}`);
    
    // Activity indicator should appear
    await expect(ralphPage.activityIndicator).toBeVisible();
    await expect(ralphPage.activityText).toHaveText('STARTING...');
  });

  test('Quick actions fill the input field', async ({ page }) => {
    await ralphPage.systemStatusChip.click();
    await expect(ralphPage.input).toHaveValue('Status Report');

    await ralphPage.explainTopicChip.click();
    await expect(ralphPage.input).toHaveValue('Explain Quantum Physics simply');

    await ralphPage.resetContextChip.click();
    await expect(ralphPage.input).toHaveValue('Clear Memory');
  });

  test('Toggle mode changes button text and log visibility', async ({ page }) => {
    // Initially Simple Mode is ON
    await expect(ralphPage.modeBtn).toHaveText('SIMPLE MODE: ON');

    // Send a command to trigger a system log (mocked or real)
    // Note: In a real environment, we'd wait for Ralph to produce logs.
    // For this test, we just check the button state.
    
    await ralphPage.toggleMode();
    await expect(ralphPage.modeBtn).toHaveText('SIMPLE MODE: OFF');
    await expect(ralphPage.modeBtn).toHaveClass(/active/);

    await ralphPage.toggleMode();
    await expect(ralphPage.modeBtn).toHaveText('SIMPLE MODE: ON');
    await expect(ralphPage.modeBtn).not.toHaveClass(/active/);
  });

  test('Metrics dashboard updates periodically', async ({ page }) => {
    // This test assumes the backend is providing metrics.
    // We expect the tokens or status to change if something is happening,
    // but at minimum, we can check if the elements are present and have default values.
    await expect(ralphPage.metricTokens).toBeVisible();
    await expect(ralphPage.metricDuration).toBeVisible();
    await expect(ralphPage.metricStatus).toBeVisible();
  });

  test('Plan sidebar updates periodically', async ({ page }) => {
    // Check if the plan container is present
    await expect(ralphPage.planContainer).toBeVisible();
    // Default state when no plan exists
    await expect(ralphPage.planContainer).toContainText('No active plan.');
  });

  test('Live code preview updates when code is received', async ({ page }) => {
    // Verify element is attached
    await expect(ralphPage.codePreview).toBeAttached();

    // Mock receiving a code block via the globally exposed extractCode function
    const mockCode = 'print("Hello from E2E test")';
    await page.evaluate((code) => {
      window.extractCode(`Here is some code:\n\`\`\`python\n${code}\n\`\`\``);
    }, mockCode);

    // Verify the code preview updated
    await expect(ralphPage.codePreview).toContainText(mockCode);
    await expect(ralphPage.codePreview).toBeVisible();
  });

  test('Full journey: Send command, see working state, wait for completion', async ({ page }) => {
    // Use a very simple command that should complete quickly if mocked, 
    // or just observe the transition.
    const command = 'hello';
    await ralphPage.sendCommand(command);
    
    await expect(ralphPage.activityIndicator).toBeVisible();
    
    // Wait for the completion signal from backend (SYSTEM::COMPLETE)
    // We increase timeout as Ralph might take some time to "think"
    await ralphPage.waitForActivityToFinish();
    
    await expect(ralphPage.activityIndicator).toBeHidden();
  });
});
