import esbuild from 'esbuild'
import cssModulesPlugin from "esbuild-css-modules-plugin"
import * as importMap from "esbuild-plugin-import-map";

const args = process.argv.slice(2);
const watch = args.includes('--watch');

// Common options for all builds
const commonOptions = {
  bundle: true,
  plugins: [cssModulesPlugin()],
  minify: !watch,
  sourcemap: watch,
};

// Widget build (ESM format for AnyWidget and imports)
const widgetConfig = {
  ...commonOptions,
  format: 'esm',
  entryPoints: ['src/colight/js/widget.jsx'],
  outfile: 'src/colight/dist/widget.mjs',
  // Remove cssModulesPlugin and handle CSS inline
  plugins: [],
};

// Widget build (ESM format with .js extension for compatibility)
const widgetJsConfig = {
  ...commonOptions,
  format: 'esm',
  entryPoints: ['src/colight/js/widget.jsx'],
  outfile: 'src/colight/dist/widget.js',
  // Remove cssModulesPlugin and handle CSS inline
  plugins: [],
};

// Embed build (IIFE format for standalone use with script tags)
const embedConfig = {
  ...commonOptions,
  format: 'iife',
  globalName: 'colight', // Makes it available as window.colight
  entryPoints: ['src/colight/js/embed.js'],
  outfile: 'src/colight/dist/embed.js',
  // Remove cssModulesPlugin and handle CSS inline
  plugins: [],
};

// Apply CDN imports if enabled
const USE_CDN_IMPORTS = false //!watch
if (USE_CDN_IMPORTS) {
  importMap.load('src/colight/js/import-map.cdn.json');
  // Add to all build configurations
  [widgetConfig, widgetJsConfig, embedConfig].forEach(config => {
    config.plugins.push(importMap.plugin());
  });
}

async function runBuild() {
  try {

    if (watch) {
      // Watch mode - create contexts for all builds
      const widgetContext = await esbuild.context(widgetConfig);
      const widgetJsContext = await esbuild.context(widgetJsConfig);
      const embedContext = await esbuild.context(embedConfig);

      // Start watching
      await Promise.all([
        widgetContext.watch(),
        widgetJsContext.watch(),
        embedContext.watch()
      ]);

      console.log('Watching for changes...');
    } else {
      // Build once
      await Promise.all([
        esbuild.build(widgetConfig),
        esbuild.build(widgetJsConfig),
        esbuild.build(embedConfig)
      ]);

      console.log('Build completed successfully');
      console.log('Output files:');
      console.log(' - ' + widgetConfig.outfile);
      console.log(' - ' + widgetJsConfig.outfile);
      console.log(' - ' + embedConfig.outfile);
    }
  } catch (error) {
    console.error('Build failed:', error);
    process.exit(1);
  }
}

runBuild();
