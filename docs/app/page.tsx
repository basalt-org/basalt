import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';

async function getCSS(): Promise<string> {
  const response = await fetch('https://raw.githubusercontent.com/sindresorhus/github-markdown-css/gh-pages/github-markdown.css');
  return response.text();
}

async function getReadme(): Promise<string> {
  const response = await fetch('https://raw.githubusercontent.com/basalt-org/basalt/main/README.md');
  return response.text();
}

export default async function Home() {
  const css = await getCSS();
  const readme = await getReadme();

  return (
    <main>
      <article className="markdown-body p-20">
        <style dangerouslySetInnerHTML={{ __html: css }} />
        <ReactMarkdown rehypePlugins={[rehypeRaw]}>{readme}</ReactMarkdown>
      </article>
    </main>
  );
}
