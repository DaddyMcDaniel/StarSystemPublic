const { chromium } = require('playwright');

const USER = process.env.GITHUB_USER || 'DaddyMcDaniel';
const PASS = process.env.GITHUB_PASSWORD || 'Github123456.0';

console.log('Opening GitHub to create access token...');

(async () => {
  const browser = await chromium.launch({ 
    headless: false,
    slowMo: 500
  });
  
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  
  try {
    // Login first
    console.log('Logging into GitHub...');
    await page.goto('https://github.com/login');
    
    await page.getByLabel('Username or email address').fill(USER);
    await page.getByLabel('Password').fill(PASS);
    await page.getByRole('button', { name: 'Sign in', exact: true }).first().click();

    // Wait for potential 2FA
    await page.waitForTimeout(3000);
    const needsVerification = await page.getByText('Verify').first().isVisible().catch(() => false);
    if (needsVerification) {
      console.log('Please complete 2FA verification...');
      await page.waitForTimeout(30000); // Wait for 2FA completion
    }

    // Navigate directly to token creation
    console.log('Navigating to token creation page...');
    await page.goto('https://github.com/settings/tokens/new');
    
    // Pre-fill token settings
    await page.getByLabel('Note').fill('StarSystem Git Push Token');
    
    // Select repo scope (essential for git push)
    const repoCheckbox = page.locator('input[name="scopes"][value="repo"]');
    await repoCheckbox.check();
    
    console.log('Token creation page opened with repo scope selected.');
    console.log('Please:');
    console.log('1. Set expiration as needed');
    console.log('2. Click "Generate token"');
    console.log('3. Copy the generated token');
    console.log('4. Use the token with: git push https://DaddyMcDaniel:<token>@github.com/DaddyMcDaniel/StarSystem.git master');
    
    // Keep browser open for manual token creation
    await page.waitForTimeout(180000); // 3 minutes
    
  } catch (e) {
    console.error('Error:', e.message);
  } finally {
    await browser.close();
  }
})();