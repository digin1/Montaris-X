import {themes as prismThemes} from 'prism-react-renderer';

const siteUrl = 'https://digindominic.me';
const baseUrl = '/Montaris-X/';
const canonicalUrl = `${siteUrl}${baseUrl}`;
const siteDescription =
  'Montaris-X is a free, cross-platform desktop ROI editor for microscopy, histology, and fluorescence imaging. Draw, refine, and export annotations locally.';
const socialImage = `${canonicalUrl}img/montaris-x-social-card.png`;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Montaris-X',
  titleDelimiter: '|',
  tagline: 'Scientific ROI editor for microscopy, histology, and fluorescence imaging',
  favicon: 'img/logo.png',

  future: {
    v4: true,
  },

  url: siteUrl,
  baseUrl,

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
        '@graph': [
          {
            '@type': 'WebSite',
            '@id': `${canonicalUrl}#website`,
            url: canonicalUrl,
            name: 'Montaris-X',
            description: siteDescription,
            inLanguage: 'en',
            publisher: {'@id': `${canonicalUrl}#organization`},
          },
          {
            '@type': 'Organization',
            '@id': `${canonicalUrl}#organization`,
            name: 'Montaris-X Contributors',
            url: canonicalUrl,
            logo: {
              '@type': 'ImageObject',
              url: `${canonicalUrl}img/logo.png`,
            },
            sameAs: [
              'https://github.com/digin1/Montaris-X',
              'https://pypi.org/project/montaris-x/',
            ],
          },
          {
            '@type': 'SoftwareApplication',
            '@id': `${canonicalUrl}#software`,
            name: 'Montaris-X',
            applicationCategory: 'ScienceApplication',
            operatingSystem: 'Windows, macOS, Linux',
            description: siteDescription,
            url: canonicalUrl,
            downloadUrl: 'https://pypi.org/project/montaris-x/',
            softwareHelp: `${canonicalUrl}docs/getting-started/installation`,
            screenshot: `${canonicalUrl}img/demo.gif`,
            softwareVersion: '2.0.0',
            license: 'https://opensource.org/licenses/MIT',
            isAccessibleForFree: true,
            offers: {
              '@type': 'Offer',
              price: '0',
              priceCurrency: 'USD',
            },
            featureList: [
              'ROI editing for microscopy and histology images',
              'ImageJ .roi and ZIP import/export',
              'Multi-channel image viewing',
              'Component-aware ROI transforms',
              'Offline desktop workflow',
            ],
            publisher: {'@id': `${canonicalUrl}#organization`},
          },
        ],
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
      image: 'img/montaris-x-social-card.png',
      metadata: [
        {
          name: 'keywords',
          content:
            'scientific ROI editor, microscopy annotation software, histology ROI tool, fluorescence image annotation, imagej alternative, qupath alternative, napari alternative, scientific image annotation, brain delineation software',
        },
        {
          name: 'description',
          content: siteDescription,
        },
        {name: 'author', content: 'Montaris-X Contributors'},
        {name: 'robots', content: 'index,follow,max-image-preview:large'},
        {name: 'theme-color', content: '#0f7282'},
        {property: 'og:type', content: 'website'},
        {property: 'og:site_name', content: 'Montaris-X'},
        {
          property: 'og:title',
          content:
            'Montaris-X | Scientific ROI Editor for Microscopy, Histology, and Fluorescence Imaging',
        },
        {property: 'og:description', content: siteDescription},
        {property: 'og:image', content: socialImage},
        {
          property: 'og:image:alt',
          content: 'Montaris-X scientific ROI editor social card',
        },
        {property: 'og:locale', content: 'en_US'},
        {name: 'twitter:card', content: 'summary_large_image'},
        {
          name: 'twitter:title',
          content:
            'Montaris-X | Scientific ROI Editor for Microscopy, Histology, and Fluorescence Imaging',
        },
        {name: 'twitter:description', content: siteDescription},
        {name: 'twitter:image', content: socialImage},
      ],
      navbar: {
        title: 'Montaris-X',
        hideOnScroll: true,
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
        defaultMode: 'light',
        respectPrefersColorScheme: false,
      },
    }),
};

export default config;
