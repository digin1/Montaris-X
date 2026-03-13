/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
      ],
    },
    {
      type: 'category',
      label: 'User Guide',
      collapsed: false,
      items: [
        'user-guide/drawing-tools',
        'user-guide/transform-move',
        'user-guide/layer-management',
        'user-guide/import-export',
        'user-guide/multi-channel',
        'user-guide/session-management',
        'user-guide/image-adjustments',
        'user-guide/display-modes',
      ],
    },
    'why-montaris-x',
    'keyboard-shortcuts',
    'supported-formats',
  ],
};

export default sidebars;
