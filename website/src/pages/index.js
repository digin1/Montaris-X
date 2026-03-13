import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

const features = [
  {
    icon: '\uD83C\uDFA8',
    title: '7 Drawing Tools',
    description:
      'Brush, Eraser, Rectangle, Circle, Polygon, Stamp, and Bucket Fill — each optimized for precise ROI delineation on scientific images.',
  },
  {
    icon: '\uD83D\uDD2C',
    title: 'Scientific Image Support',
    description:
      '16/32-bit TIFFs, multi-channel composites, false-color display, and tiled rendering for images of any size.',
  },
  {
    icon: '\uD83E\uDDE9',
    title: 'Component-Aware Editing',
    description:
      'Transform and move individual connected components within an ROI — no need to split layers manually.',
  },
  {
    icon: '\uD83D\uDCC2',
    title: 'ImageJ Compatible',
    description:
      'Import and export .roi files and ZIP bundles directly. Drop-in replacement for ImageJ ROI workflows.',
  },
  {
    icon: '\u26A1',
    title: 'Fast & Lightweight',
    description:
      'Native desktop performance with GPU-accelerated rendering. Handles thousands of ROIs without lag.',
  },
  {
    icon: '\uD83D\uDD12',
    title: 'Offline & Private',
    description:
      'Runs entirely on your machine. No cloud, no account, no data upload. Your research stays yours.',
  },
];

function FeatureCard({icon, title, description}) {
  return (
    <div className="featureCard">
      <span className="featureIcon">{icon}</span>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function ComparisonTable() {
  const rows = [
    ['Purpose-built for ROI delineation', 'Yes', 'No', 'No', 'No'],
    ['Native desktop performance', 'Yes', 'Yes', 'Yes', 'Partial'],
    ['16/32-bit TIFF support', 'Yes', 'Yes', 'Yes', 'Yes'],
    ['Multi-channel composites', 'Yes', 'Yes', 'Yes', 'Yes'],
    ['Component-aware transform', 'Yes', 'No', 'No', 'No'],
    ['ImageJ .roi import/export', 'Yes', 'Native', 'No', 'No'],
    ['Brush + Polygon + Stamp tools', 'Yes', 'Partial', 'Partial', 'Partial'],
    ['Session auto-save & recovery', 'Yes', 'No', 'No', 'No'],
    ['No Java runtime required', 'Yes', 'No', 'No', 'Yes'],
    ['Cross-platform (Win/Mac/Linux)', 'Yes', 'Yes', 'Yes', 'Yes'],
    ['Free & open source', 'MIT', 'Public domain', 'GPL', 'BSD'],
  ];

  return (
    <div className="comparisonTable">
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>Montaris-X</th>
            <th>ImageJ / FIJI</th>
            <th>QuPath</th>
            <th>Napari</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="ROI Editor for Scientific Microscopy Images"
      description="Montaris-X is a free, cross-platform ROI editor for scientific microscopy images. Purpose-built for brain delineation, histology annotation, and fluorescence imaging.">
      {/* Hero */}
      <header className="heroBanner">
        <div className="container">
          <img
            src="img/logo.png"
            alt="Montaris-X Logo"
            width="96"
            style={{marginBottom: '1rem'}}
          />
          <Heading as="h1" className="heroTitle">
            {siteConfig.tagline}
          </Heading>
          <p className="heroSubtitle">
            A free, cross-platform desktop application for drawing, editing, and
            managing Regions of Interest on scientific microscopy images.
          </p>
          <div className="heroButtons">
            <Link
              className="button button--primary button--lg"
              to="/docs/getting-started/installation">
              Get Started
            </Link>
            <Link
              className="button button--outline button--lg"
              to="/docs/why-montaris-x">
              Why Montaris-X?
            </Link>
          </div>
        </div>
      </header>

      <main>
        {/* Demo */}
        <section className="container sectionContainer">
          <div style={{textAlign: 'center'}}>
            <Heading as="h2">See It in Action</Heading>
            <img
              src="img/demo.gif"
              alt="Montaris-X demo showing ROI drawing on a brain section"
              className="demoImage"
            />
          </div>
        </section>

        {/* Features */}
        <section className="container sectionContainer">
          <Heading as="h2" style={{textAlign: 'center'}}>
            Built for Delineation Work
          </Heading>
          <p style={{textAlign: 'center', maxWidth: 700, margin: '0 auto 2rem'}}>
            General-purpose tools like ImageJ and Napari can annotate images, but
            they weren't designed for the specific workflow of delineating dozens
            of ROIs on large scientific images. Montaris-X is.
          </p>
          <div className="featureGrid">
            {features.map((f, i) => (
              <FeatureCard key={i} {...f} />
            ))}
          </div>
        </section>

        {/* Comparison */}
        <section className="container sectionContainer">
          <Heading as="h2" style={{textAlign: 'center'}}>
            How Montaris-X Compares
          </Heading>
          <ComparisonTable />
        </section>

        {/* CTA */}
        <section className="container">
          <div className="ctaSection">
            <Heading as="h2">Ready to Get Started?</Heading>
            <p>Install Montaris-X in seconds with pip:</p>
            <code>pip install montaris-x</code>
            <br />
            <br />
            <Link
              className="button button--secondary button--lg"
              to="/docs/getting-started/installation">
              Installation Guide
            </Link>
          </div>
        </section>
      </main>
    </Layout>
  );
}
