import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Montaris-X',
  tagline: 'The ROI Editor Built for Science',
  favicon: 'img/logo.png',

  future: {
    v4: true,
  },

  url: 'https://digindominic.me',
  baseUrl: '/Montaris-X/',

  organizationName: 'digin1',
  projectName: 'Montaris-X',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  headTags: [
    {
      tagName: 'script',
      attributes: {type: 'application/ld+json'},
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'SoftwareApplication',
        name: 'Montaris-X',
        applicationCategory: 'ScienceApplication',
        operatingSystem: 'Windows, macOS, Linux',
        offers: {price: '0', priceCurrency: 'USD'},
        description:
          'Cross-platform ROI editor for scientific microscopy images. Purpose-built for brain delineation, histology annotation, and fluorescence microscopy.',
        url: 'https://digindominic.me/Montaris-X/',
        downloadUrl: 'https://pypi.org/project/montaris-x/',
        softwareVersion: '2.0.0',
        license: 'https://opensource.org/licenses/MIT',
      }),
    },
  ],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/digin1/Montaris-X/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          lastmod: 'date',
          changefreq: 'weekly',
          priority: 0.5,
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/demo.gif',
      metadata: [
        {
          name: 'keywords',
          content:
            'roi editor, region of interest, brain delineation, neuroscience annotation, microscopy roi, imagej alternative, qupath alternative, napari alternative, scientific image annotation, histology annotation, fluorescence microscopy',
        },
        {
          name: 'description',
          content:
            'Montaris-X is a free, cross-platform ROI editor for scientific microscopy images. Purpose-built for brain delineation, histology, and fluorescence imaging.',
        },
        {property: 'og:type', content: 'website'},
        {name: 'twitter:card', content: 'summary_large_image'},
      ],
      navbar: {
        title: 'Montaris-X',
        logo: {
          alt: 'Montaris-X Logo',
          src: 'img/logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Docs',
          },
          {
            to: '/docs/why-montaris-x',
            label: 'Why Montaris-X?',
            position: 'left',
          },
          {
            href: 'https://github.com/digin1/Montaris-X',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: 'https://pypi.org/project/montaris-x/',
            label: 'PyPI',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              {label: 'Getting Started', to: '/docs/getting-started/installation'},
              {label: 'User Guide', to: '/docs/user-guide/drawing-tools'},
              {label: 'Keyboard Shortcuts', to: '/docs/keyboard-shortcuts'},
            ],
          },
          {
            title: 'Resources',
            items: [
              {label: 'Why Montaris-X?', to: '/docs/why-montaris-x'},
              {label: 'Supported Formats', to: '/docs/supported-formats'},
              {label: 'Import & Export', to: '/docs/user-guide/import-export'},
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/digin1/Montaris-X',
              },
              {
                label: 'Issues',
                href: 'https://github.com/digin1/Montaris-X/issues',
              },
              {
                label: 'PyPI',
                href: 'https://pypi.org/project/montaris-x/',
              },
            ],
          },
        ],
        copyright: `Copyright \u00A9 ${new Date().getFullYear()} Digin Dominic and Montaris-X Contributors. Released under the MIT License.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      colorMode: {
        defaultMode: 'dark',
        respectPrefersColorScheme: true,
      },
    }),
};

export default config;
