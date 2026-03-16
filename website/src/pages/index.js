import Head from '@docusaurus/Head';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import FeedbackForm from '../components/FeedbackForm';

const heroSignals = [
  'Windows, macOS, and Linux',
  '16/32-bit TIFF and multi-channel support',
  'ImageJ .roi and ZIP compatible',
  'Offline by default',
];

const metrics = [
  {value: '16/32-bit', label: 'Microscopy-ready image depth'},
  {value: '7 tools', label: 'Drawing, stamping, and fill workflows'},
  {value: '20 colors', label: 'Distinct ROI palettes for layered review'},
  {value: '0 upload', label: 'Everything stays on your machine'},
];

const capabilityCards = [
  {
    eyebrow: 'Microscopy-ready viewing',
    title: 'Read scientific images without flattening the workflow',
    description:
      'Montaris-X opens the image types research teams actually work with and keeps contrast, channel, and orientation controls close to the annotation surface.',
    points: [
      '16-bit and 32-bit TIFF support with tiled rendering for large datasets.',
      'Composite, montage, grayscale, and false-color display modes for multi-channel review.',
      'Fast brightness, gamma, and exposure adjustments while you annotate.',
    ],
  },
  {
    eyebrow: 'ROI editing',
    title: 'Delineate complex structures without layer gymnastics',
    description:
      'The editing model is tuned for repeated scientific delineation work rather than generic painting or image manipulation.',
    points: [
      'Brush, eraser, polygon, rectangle, circle, stamp, and bucket fill in one toolset.',
      'Component-aware transforms let you move or scale connected regions without splitting layers.',
      'Auto-save and session recovery reduce the risk of losing long review sessions.',
    ],
  },
  {
    eyebrow: 'Interoperability',
    title: 'Stay compatible with downstream analysis pipelines',
    description:
      'Montaris-X fits into labs that already rely on ImageJ, Python, or mask-based analysis without forcing a proprietary export path.',
    points: [
      'Import and export ImageJ .roi files and ZIP bundles directly.',
      'Save native .npz sessions with labels, colors, and metadata preserved.',
      'Export masks and overlays for reporting, QC, or post-processing steps.',
    ],
  },
  {
    eyebrow: 'Local-first delivery',
    title: 'Desktop software that respects scientific data handling',
    description:
      'The application runs locally, keeps research images on your machine, and ships as an open-source project that teams can inspect and extend.',
    points: [
      'No mandatory account, server dependency, or cloud upload step.',
      'Cross-platform desktop delivery for Windows, macOS, and Linux labs.',
      'MIT-licensed source on GitHub with PyPI distribution for direct install.',
    ],
  },
];

const workflowSteps = [
  {
    step: '01',
    title: 'Load microscopy or histology images',
    description:
      'Open fluorescence, brightfield, atlas, or section images and keep scientific display controls close to the canvas.',
  },
  {
    step: '02',
    title: 'Build layered ROI sets quickly',
    description:
      'Create distinct ROI layers, assign readable colors, and switch tools based on the structure you need to capture.',
  },
  {
    step: '03',
    title: 'Refine components with precision',
    description:
      'Tighten masks, transform connected components, and compare channels without leaving the editing surface.',
  },
  {
    step: '04',
    title: 'Export to the format the lab expects',
    description:
      'Ship ImageJ ROI bundles, masks, overlays, or native sessions with minimal cleanup before analysis.',
  },
];

const comparisonRows = [
  ['Purpose-built for ROI delineation', 'Yes', 'No', 'No', 'No'],
  ['16/32-bit TIFF support', 'Yes', 'Yes', 'Yes', 'Yes'],
  ['Multi-channel composite viewing', 'Yes', 'Yes', 'Yes', 'Yes'],
  ['Component-aware transform', 'Yes', 'No', 'No', 'No'],
  ['ImageJ .roi import/export', 'Yes', 'Native', 'Yes', 'Plugin'],
  ['Brush, polygon, and stamp tools', 'Yes', 'Partial', 'Partial', 'Partial'],
  ['Session auto-save and recovery', 'Yes', 'No', 'No', 'No'],
  ['Cross-platform desktop app', 'Yes', 'Yes', 'Yes', 'Yes'],
  ['Free and open source', 'MIT', 'Public domain / GPL', 'GPL', 'BSD'],
];

const faqItems = [
  {
    question: 'Who is Montaris-X built for?',
    answer:
      'It is aimed at research teams working on microscopy, histology, atlas delineation, fluorescence imaging, and other scientific annotation workflows where ROI precision matters.',
  },
  {
    question: 'Can I keep using existing ImageJ ROI files?',
    answer:
      'Yes. Montaris-X imports and exports ImageJ .roi files and ZIP bundles so labs can adopt it without rebuilding established ROI archives.',
  },
  {
    question: 'Does it require an online account or hosted backend?',
    answer:
      'No. The editor runs locally on your machine, which keeps research data private and makes the workflow easier to deploy in controlled environments.',
  },
  {
    question: 'Is it limited to one image style or one operating system?',
    answer:
      'No. It is designed for scientific image review across Windows, macOS, and Linux, including large TIFF, fluorescence, and multi-channel image workflows.',
  },
];

function CapabilityCard({eyebrow, title, description, points}) {
  return (
    <article className="scienceCard">
      <span className="scienceCard__eyebrow">{eyebrow}</span>
      <h3>{title}</h3>
      <p>{description}</p>
      <ul>
        {points.map((point) => (
          <li key={point}>{point}</li>
        ))}
      </ul>
    </article>
  );
}

function WorkflowCard({step, title, description}) {
  return (
    <article className="workflowCard">
      <span className="workflowCard__step">{step}</span>
      <h3>{title}</h3>
      <p>{description}</p>
    </article>
  );
}

function ComparisonCell({value}) {
  const normalizedValue = value.toLowerCase();
  let tone = 'neutral';

  if (normalizedValue === 'yes' || normalizedValue === 'native' || normalizedValue === 'mit') {
    tone = 'positive';
  }

  if (normalizedValue === 'no') {
    tone = 'negative';
  }

  return <span className={`comparisonChip comparisonChip--${tone}`}>{value}</span>;
}

function ComparisonTable() {
  return (
    <div className="comparisonPanel">
      <div className="comparisonTable">
        <table>
          <thead>
            <tr>
              <th>Capability</th>
              <th>Montaris-X</th>
              <th>ImageJ / FIJI</th>
              <th>QuPath</th>
              <th>Napari</th>
            </tr>
          </thead>
          <tbody>
            {comparisonRows.map((row) => (
              <tr key={row[0]}>
                <td>{row[0]}</td>
                {row.slice(1).map((cell) => (
                  <td key={`${row[0]}-${cell}`}>
                    <ComparisonCell value={cell} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FAQCard({question, answer}) {
  return (
    <article className="faqCard">
      <h3>{question}</h3>
      <p>{answer}</p>
    </article>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  const siteUrl = `${siteConfig.url}${siteConfig.baseUrl}`;
  const pageTitle =
    'Scientific ROI Editor for Microscopy, Histology, and Fluorescence Imaging';
  const pageDescription =
    'Montaris-X is a free, cross-platform desktop ROI editor for microscopy, histology, and fluorescence imaging. Draw, refine, compare, and export annotations without leaving your local machine.';
  const pageImage = `${siteUrl}img/montaris-x-social-card.png`;
  const faqSchema = {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: faqItems.map((item) => ({
      '@type': 'Question',
      name: item.question,
      acceptedAnswer: {
        '@type': 'Answer',
        text: item.answer,
      },
    })),
  };

  return (
    <Layout
      title={pageTitle}
      description={pageDescription}
      wrapperClassName="sciencePage">
      <Head>
        <link rel="canonical" href={siteUrl} />
        <meta
          name="keywords"
          content="scientific ROI editor, microscopy annotation software, histology ROI tool, fluorescence image annotation, ImageJ alternative, QuPath alternative, Napari alternative"
        />
        <meta name="robots" content="index,follow,max-image-preview:large" />
        <meta property="og:title" content={`Montaris-X | ${pageTitle}`} />
        <meta property="og:description" content={pageDescription} />
        <meta property="og:image" content={pageImage} />
        <meta
          property="og:image:alt"
          content="Montaris-X scientific ROI editor social card"
        />
        <meta name="twitter:title" content={`Montaris-X | ${pageTitle}`} />
        <meta name="twitter:description" content={pageDescription} />
        <meta name="twitter:image" content={pageImage} />
        <meta
          name="twitter:image:alt"
          content="Montaris-X scientific ROI editor social card"
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{__html: JSON.stringify(faqSchema)}}
        />
      </Head>

      <header className="scienceHero">
        <div className="container scienceHero__layout">
          <div className="scienceHero__content">
            <span className="scienceEyebrow">Scientific ROI annotation software</span>
            <Heading as="h1" className="scienceHero__title">
              Precision ROI editing for microscopy, histology, and fluorescence imaging.
            </Heading>
            <p className="scienceHero__subtitle">
              Montaris-X is a free desktop ROI editor built for labs that need
              accurate delineation, multi-channel viewing, and clean exports
              without cloud lock-in or heavyweight setup.
            </p>

            <div className="scienceHero__actions">
              <Link
                className="button button--sciencePrimary button--lg"
                to="/docs/getting-started/installation">
                Install Montaris-X
              </Link>
              <Link
                className="button button--scienceGhost button--lg"
                to="/docs/why-montaris-x">
                Compare the workflow
              </Link>
            </div>

            <div className="sciencePillRow">
              {heroSignals.map((signal) => (
                <span key={signal} className="sciencePill">
                  {signal}
                </span>
              ))}
            </div>
          </div>

          <div className="scienceHero__panel">
            <div className="sciencePanel">
              <div className="sciencePanel__header">
                <span className="sciencePanel__label">Built for lab image review</span>
                <span className="sciencePanel__status">Local-first desktop app</span>
              </div>

              <img
                src="img/demo.gif"
                alt="Montaris-X interface showing ROI drawing on a scientific image"
                className="sciencePanel__image"
                loading="lazy"
                width="800"
                height="500"
              />

              <div className="scienceMetricGrid">
                {metrics.map((metric) => (
                  <div key={metric.label} className="scienceMetric">
                    <strong>{metric.value}</strong>
                    <span>{metric.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main>
        <section className="container scienceSection">
          <div className="sectionIntro">
            <span className="sectionIntro__eyebrow">Why it stands out</span>
            <Heading as="h2">Purpose-built for scientific delineation work</Heading>
            <p>
              Generic image viewers can draw masks, but they are usually not
              optimized for repeated ROI review across large microscopy or
              histology images. Montaris-X is tuned for that exact task.
            </p>
          </div>

          <div className="scienceCardGrid scienceCardGrid--wide">
            {capabilityCards.map((card) => (
              <CapabilityCard key={card.title} {...card} />
            ))}
          </div>
        </section>

        <section className="container scienceSection">
          <div className="sectionIntro">
            <span className="sectionIntro__eyebrow">Workflow</span>
            <Heading as="h2">Move from image import to export without tool sprawl</Heading>
            <p>
              The product flow stays focused: load a scientific image, build ROI
              layers, refine structures, and export to the format your pipeline
              already expects.
            </p>
          </div>

          <div className="workflowGrid">
            {workflowSteps.map((step) => (
              <WorkflowCard key={step.step} {...step} />
            ))}
          </div>
        </section>

        <section className="container scienceSection">
          <div className="sectionIntro">
            <span className="sectionIntro__eyebrow">Comparison</span>
            <Heading as="h2">A tighter fit than general-purpose image tools</Heading>
            <p>
              Montaris-X keeps ROI work fast and focused while remaining
              compatible with established ImageJ workflows and common scientific
              image formats.
            </p>
          </div>

          <ComparisonTable />
        </section>

        <section className="container scienceSection">
          <div className="sectionIntro">
            <span className="sectionIntro__eyebrow">FAQ</span>
            <Heading as="h2">Common questions from research teams</Heading>
            <p>
              These are the questions most likely to come up when evaluating an
              ROI editor for scientific image annotation.
            </p>
          </div>

          <div className="faqGrid">
            {faqItems.map((item) => (
              <FAQCard key={item.question} {...item} />
            ))}
          </div>
        </section>

        <section className="container scienceSection">
          <div className="sectionIntro">
            <span className="sectionIntro__eyebrow">Feedback</span>
            <Heading as="h2">Share your thoughts with the team</Heading>
            <p>
              Have a feature idea, found a bug, or want to share how you use
              Montaris-X? Your feedback is submitted as a GitHub issue so the
              community can track and discuss it.
            </p>
          </div>

          <FeedbackForm />
        </section>

        <section className="container scienceSection">
          <div className="ctaBand">
            <div>
              <span className="sectionIntro__eyebrow sectionIntro__eyebrow--light">
                Start quickly
              </span>
              <Heading as="h2">Install once, annotate locally, export anywhere.</Heading>
              <p>
                Use the installation guide for platform-specific setup, or jump
                straight in from PyPI if you already manage Python environments.
              </p>
            </div>

            <div className="ctaBand__actions">
              <code>pip install montaris-x</code>
              <div className="ctaBand__buttons">
                <Link
                  className="button button--inverse button--lg"
                  to="/docs/getting-started/installation">
                  Read installation guide
                </Link>
                <Link
                  className="button button--ghostLight button--lg"
                  to="/docs/getting-started/quick-start">
                  Open quick start
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
