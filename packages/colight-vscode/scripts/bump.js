const fs = require('fs');
const path = require('path');
const semver = require('semver');

function getCurrentVersion() {
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  return packageJson.version;
}

function getNewVersion() {
  const currentVersion = getCurrentVersion();
  return semver.inc(currentVersion, 'patch');
}

function updatePackageJson(newVersion) {
  const packageJsonPath = path.join(__dirname, '..', 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  packageJson.version = newVersion;
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
}

function updateChangelog(newVersion) {
  const changelogPath = path.join(__dirname, '..', 'CHANGELOG.md');
  const changelog = fs.readFileSync(changelogPath, 'utf8');
  const date = new Date().toISOString().split('T')[0];
  
  // Split the changelog into sections
  const [header, unreleasedSection, ...rest] = changelog.split('## [Unreleased]');
  
  // Create the new version entry with the content from the Unreleased section
  const newEntry = `## [${newVersion}] - ${date}\n\n${unreleasedSection.trim()}\n\n`;
  
  // Create a new Unreleased section
  const newUnreleased = `## [Unreleased]\n\n### Added\n\n### Changed\n\n### Fixed\n\n`;
  
  // Combine all parts
  const updatedChangelog = `${header}${newUnreleased}${newEntry}${rest.join('## [Unreleased]')}`;
  
  fs.writeFileSync(changelogPath, updatedChangelog);
}

function main() {
  const newVersion = getNewVersion();
  updatePackageJson(newVersion);
  updateChangelog(newVersion);
  console.log(`Version bumped to ${newVersion}`);
}

if (require.main === module) {
  main();
}