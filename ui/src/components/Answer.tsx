import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// Model output is rendered as React elements, never raw HTML.
export default function Answer({ text }: { text: string }) {
  return (
    <div className="answer">
      <Markdown remarkPlugins={[remarkGfm]}>{text}</Markdown>
    </div>
  )
}
