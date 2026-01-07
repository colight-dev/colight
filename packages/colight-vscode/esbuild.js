const esbuild = require("esbuild");
const fs = require("fs");
const path = require("path");

const production = process.argv.includes('--production');
const watch = process.argv.includes('--watch');

// Copy widget.mjs from colight package to media folder
function copyWidgetAssets() {
	const srcDir = path.join(__dirname, '..', 'colight', 'src', 'colight', 'js-dist');
	const destDir = path.join(__dirname, 'media');

	// Ensure media directory exists
	if (!fs.existsSync(destDir)) {
		fs.mkdirSync(destDir, { recursive: true });
	}

	// Copy widget.mjs
	const widgetSrc = path.join(srcDir, 'widget.mjs');
	const widgetDest = path.join(destDir, 'widget.mjs');

	if (fs.existsSync(widgetSrc)) {
		fs.copyFileSync(widgetSrc, widgetDest);
		console.log('Copied widget.mjs to media/');
	} else {
		console.warn('Warning: widget.mjs not found at', widgetSrc);
		console.warn('Run "yarn build" in packages/colight first');
	}
}

/**
 * @type {import('esbuild').Plugin}
 */
const esbuildProblemMatcherPlugin = {
	name: 'esbuild-problem-matcher',

	setup(build) {
		build.onStart(() => {
			console.log('[watch] build started');
		});
		build.onEnd((result) => {
			result.errors.forEach(({ text, location }) => {
				console.error(`✘ [ERROR] ${text}`);
				console.error(`    ${location.file}:${location.line}:${location.column}:`);
			});
			console.log('[watch] build finished');
		});
	},
};

async function main() {
	// Copy widget assets before building
	copyWidgetAssets();

	const ctx = await esbuild.context({
		entryPoints: [
			'src/extension.ts'
		],
		bundle: true,
		format: 'cjs',
		minify: production,
		sourcemap: !production,
		sourcesContent: false,
		platform: 'node',
		outfile: 'dist/extension.js',
		external: ['vscode'],
		logLevel: 'silent',
		plugins: [
			/* add to the end of plugins array */
			esbuildProblemMatcherPlugin,
		],
	});
	if (watch) {
		await ctx.watch();
	} else {
		await ctx.rebuild();
		await ctx.dispose();
	}
}

main().catch(e => {
	console.error(e);
	process.exit(1);
});
