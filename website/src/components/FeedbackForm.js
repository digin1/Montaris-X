import {useState} from 'react';

const feedbackCategories = [
  'General feedback',
  'Feature request',
  'Bug report',
  'Documentation',
  'Other',
];

export default function FeedbackForm() {
  const [category, setCategory] = useState(feedbackCategories[0]);
  const [message, setMessage] = useState('');
  const [submitted, setSubmitted] = useState(false);

  function handleSubmit(e) {
    e.preventDefault();
    const title = encodeURIComponent(`[Feedback] ${category}`);
    const body = encodeURIComponent(
      `**Category:** ${category}\n\n**Feedback:**\n${message}`,
    );
    window.open(
      `https://github.com/digin1/Montaris-X/issues/new?title=${title}&body=${body}&labels=feedback`,
      '_blank',
      'noopener',
    );
    setSubmitted(true);
  }

  if (submitted) {
    return (
      <div className="feedbackForm feedbackForm--done">
        <p className="feedbackForm__thanks">
          Thank you for your feedback! A GitHub issue tab has been opened for you
          to finalize and submit.
        </p>
        <button
          type="button"
          className="button button--scienceGhost"
          onClick={() => {
            setCategory(feedbackCategories[0]);
            setMessage('');
            setSubmitted(false);
          }}>
          Send another
        </button>
      </div>
    );
  }

  return (
    <form className="feedbackForm" onSubmit={handleSubmit}>
      <label className="feedbackForm__label" htmlFor="feedback-category">
        Category
      </label>
      <select
        id="feedback-category"
        className="feedbackForm__select"
        value={category}
        onChange={(e) => setCategory(e.target.value)}>
        {feedbackCategories.map((cat) => (
          <option key={cat} value={cat}>
            {cat}
          </option>
        ))}
      </select>

      <label className="feedbackForm__label" htmlFor="feedback-message">
        Message
      </label>
      <textarea
        id="feedback-message"
        className="feedbackForm__textarea"
        rows={4}
        placeholder="Tell us what you think, request a feature, or report a bug…"
        required
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      />

      <button type="submit" className="button button--sciencePrimary">
        Submit via GitHub
      </button>
    </form>
  );
}
