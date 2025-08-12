import { chromium } from 'playwright';
const USER = process.env.GITHUB_USER || '';
const PASS = process.env.GITHUB_PASSWORD || '';
const REPO = process.env.REPO_NAME || 'starsystem';
const VIS  = process.env.REPO_VISIBILITY || 'private'; // 'public' | 'private'
if (!USER || !PASS) {
  console.error(JSON.stringify({ success:false, error:{code:"missing_env", message:"GITHUB_USER/GITHUB_PASSWORD required for UI login"} }));
  process.exit(0);
}
(async () => {
  const browser = await chromium.launch({ headless: false }); // allow 2FA
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  try {
    await page.goto('https://github.com/login');
    await page.getByLabel('Username or email address').fill(USER);
    await page.getByLabel('Password').fill(PASS);
    await page.getByRole('button', { name: 'Sign in' }).click();

    // If 2FA present, wait up to 120s for user to complete.
    if (await page.getByText('Verify').first().isVisible().catch(() => false)) {
      await page.waitForTimeout(120000);
    }

    await page.goto('https://github.com/new');
    await page.getByLabel('Repository name').fill(REPO);
    if (VIS === 'private') {
      await page.getByLabel('Private').check({ force:true });
    } else {
      await page.getByLabel('Public').check({ force:true });
    }
    // Initialize with README to simplify first push
    const initReadme = await page.getByLabel('Add a README file').isVisible().catch(() => false);
    if (initReadme) await page.getByLabel('Add a README file').check({ force:true });

    await page.getByRole('button', { name: 'Create repository' }).click();
    await page.waitForURL(/github\.com\/.+?\/.+?$/);

    const url = page.url();
    await page.screenshot({ path: 'out/repo_created.png', fullPage: true });
    console.log(JSON.stringify({ success:true, data:{ repo_url:url } }));
  } catch (e:any) {
    console.error(JSON.stringify({ success:false, error:{ code:"playwright_login_failed", message:e?.message || String(e) } }));
  } finally {
    await ctx.close(); await browser.close();
  }
})();