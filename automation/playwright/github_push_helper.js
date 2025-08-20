const { chromium } = require('playwright');

const USER = process.env.GITHUB_USER || 'DaddyMcDaniel';
const PASS = process.env.GITHUB_PASSWORD || 'Github123456.0';
const REPO_URL = 'https://github.com/DaddyMcDaniel/StarSystem';

console.log('Starting GitHub authentication helper...');

(async () => {
  const browser = await chromium.launch({ 
    headless: false,  // Keep visible so user can complete 2FA if needed
    slowMo: 1000     // Slow down actions for visibility
  });
  
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  
  try {
    console.log('Navigating to GitHub login...');
    await page.goto('https://github.com/login');
    
    console.log('Filling login credentials...');
    await page.getByLabel('Username or email address').fill(USER);
    await page.getByLabel('Password').fill(PASS);
    
    console.log('Clicking sign in...');
    await page.getByRole('button', { name: 'Sign in', exact: true }).first().click();

    // Check for 2FA or other verification
    console.log('Checking for 2FA or verification...');
    try {
      await page.waitForTimeout(3000);
      
      // Check if we need 2FA
      const needsVerification = await page.getByText('Verify').first().isVisible().catch(() => false);
      if (needsVerification) {
        console.log('2FA detected. Please complete verification in the browser...');
        console.log('Waiting up to 2 minutes for completion...');
        await page.waitForTimeout(120000);
      }
      
      // Check if login was successful by looking for the GitHub dashboard
      await page.waitForURL(/github\.com/, { timeout: 30000 });
      console.log('Login successful!');
      
    } catch (e) {
      console.log('Authentication may have failed or timed out:', e.message);
    }

    // Navigate to the repository
    console.log(`Navigating to repository: ${REPO_URL}`);
    await page.goto(REPO_URL);
    
    // Navigate to settings to check access tokens or repository settings
    console.log('Navigating to repository settings...');
    await page.goto(`${REPO_URL}/settings`);
    
    // Take a screenshot for debugging
    await page.screenshot({ path: 'out/github_repo_settings.png', fullPage: true });
    console.log('Screenshot saved to out/github_repo_settings.png');
    
    // Keep browser open for manual token generation if needed
    console.log('Browser will remain open. You can:');
    console.log('1. Go to Settings > Developer settings > Personal access tokens');
    console.log('2. Generate a new token with repo permissions');
    console.log('3. Use that token for git authentication');
    console.log('Press Ctrl+C to close when done.');
    
    // Keep the browser open until user intervention
    await page.waitForTimeout(300000); // 5 minutes
    
  } catch (e) {
    console.error('Error during GitHub authentication:', e.message);
    await page.screenshot({ path: 'out/github_auth_error.png', fullPage: true });
  } finally {
    console.log('Closing browser...');
    await ctx.close();
    await browser.close();
  }
})();