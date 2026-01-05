const fs = require('fs');
const { execSync } = require('child_process');

function getCurrentVersion() {
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  return packageJson.version;
}
function gitCommitAndTag() {
  const version = getCurrentVersion();
  try {
    execSync('git add CHANGELOG.md package.json');
    execSync(`git commit -m "Release version ${version}"`);
    execSync(`git tag -a v${version} -m "Release version ${version}"`);
    console.log(`Version ${version} has been committed and tagged.`);
  } catch (error) {
    console.error('Error creating git commit and tag:', error.message);
    process.exit(1);
  }
}

function main() {
  gitCommitAndTag();
}

if (require.main === module) {
  main();
}