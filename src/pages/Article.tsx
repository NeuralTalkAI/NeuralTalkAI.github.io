import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkDirective from 'remark-directive'
import rehypeKatex from 'rehype-katex'
import remarkMdniceBlocks, { normalizeMdniceContainers, remarkUnswallowMath } from '../markdown/remarkMdniceBlocks'
import rehypeYamabuki from '../markdown/rehypeYamabuki'
import { getArticleToc, normalizeArticleImageUrls, remarkArticleHeadingIds, stripManualTocList, type TocItem } from '../markdown/articleStructure'
import 'katex/dist/katex.min.css'
import '../markdown/yamabuki.css'
import { ArrowLeft, ArrowRight, ArrowUpRight, MessageCircle } from 'lucide-react'
import { loadPostContent, posts } from '../data/posts'

const authorName = 'NeuralTalk'
const authorMark = 'NT'

function formatDate(date: string) {
  const [year, month, day] = date.split('-')
  return `${year} 年 ${Number(month)} 月 ${Number(day)} 日`
}

function getReadingTime(content: string) {
  const cjk = (content.match(/[一-鿿]/g) || []).length
  const words = (content.replace(/[一-鿿]/g, ' ').match(/[A-Za-z0-9]+/g) || []).length
  return Math.max(1, Math.ceil((cjk + words / 2) / 320))
}

export function ArticleNotFound() {
  return <main className="page-shell narrow-page not-found"><p className="section-kicker">404</p><h1>没有找到这篇文章</h1><p>文章可能已移动或尚未收录。</p><Link to="/" className="primary-link"><ArrowLeft size={17} />返回首页</Link></main>
}

function ArticleToc({ items }: { items: TocItem[] }) {
  const groups: { chapter: TocItem; sections: TocItem[] }[] = []
  for (const item of items) {
    if (item.depth === 2) groups.push({ chapter: item, sections: [] })
    else if (groups.length) groups[groups.length - 1].sections.push(item)
  }
  const renderLink = (item: TocItem) => (
    <a href={`#${item.id}`} onClick={(event) => {
      event.preventDefault()
      document.getElementById(item.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }}>{item.text}</a>
  )
  return (
    <nav className="article-toc" aria-label="文章章节目录">
      <ul>
        {groups.map(({ chapter, sections }) => (
          <li key={chapter.id}>
            {renderLink(chapter)}
            {sections.length > 0 && <ul>{sections.map((section) => <li key={section.id}>{renderLink(section)}</li>)}</ul>}
          </li>
        ))}
      </ul>
    </nav>
  )
}

function Utterances({ title }: { title: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    container.innerHTML = ''
    const script = document.createElement('script')
    script.src = 'https://utteranc.es/client.js'
    script.setAttribute('repo', 'NeuralTalkAI/NeuralTalkAI.github.io')
    script.setAttribute('issue-term', 'title')
    script.setAttribute('theme', 'github-light')
    script.crossOrigin = 'anonymous'
    script.async = true
    container.appendChild(script)
  }, [title])
  return <div className="utterances-container" ref={containerRef} />
}

export default function Article() {
  const { slug } = useParams()
  const post = posts.find((item) => item.slug === slug)
  const [content, setContent] = useState<string | null>(post?.content ?? null)
  useEffect(() => { window.scrollTo(0, 0) }, [slug])
  useEffect(() => {
    document.title = post ? `${post.title} | NeuralTalk` : 'NeuralTalk'
    return () => { document.title = 'NeuralTalk' }
  }, [post])
  useEffect(() => {
    let alive = true
    const embedded = posts.find((item) => item.slug === slug)?.content
    setContent(embedded ?? null)
    if (!embedded) {
      loadPostContent(slug ?? '').then((loaded) => {
        if (alive) setContent(loaded ?? '')
      })
    }
    return () => { alive = false }
  }, [slug])
  const currentIndex = posts.findIndex((item) => item.slug === slug)
  const nextPost = currentIndex >= 0 ? posts[currentIndex + 1] : undefined
  // 预加载下一篇，减少翻页等待
  useEffect(() => {
    if (nextPost) loadPostContent(nextPost.slug)
  }, [nextPost])
  if (!post) return <ArticleNotFound />
  if (content === null) {
    return <main className="article-shell"><article><header className="article-header"><Link to="/" className="back-link"><ArrowLeft size={16} />返回文章列表</Link><h1>{post.title}</h1><p className="loading-hint">文章加载中…</p></header></article></main>
  }
  const toc = getArticleToc(content)
  const tocIndex = content.indexOf('## 本文目录')
  const headingsBeforeToc = tocIndex >= 0 ? content.slice(0, tocIndex).match(/^#{1,6}\s+/gm)?.length ?? 0 : -1
  const tocHeadingId = headingsBeforeToc >= 0 ? `article-heading-${headingsBeforeToc}` : null
  const markdown = normalizeArticleImageUrls(stripManualTocList(normalizeMdniceContainers(content)))
  return (
    <main className="article-shell">
      <article>
        <header className="article-header">
          <Link to="/" className="back-link"><ArrowLeft size={16} />返回文章列表</Link>
          <div className="post-meta"><span>{post.category}</span>{post.original && <span className="original-badge">原创</span>}</div>
          <h1>{post.title}</h1>
          <div className="article-byline"><span className="mini-avatar">{authorMark}</span><span>{authorName}</span><span className="meta-dot">·</span><time>{formatDate(post.date)}</time><span className="meta-dot">·</span><span>约 {getReadingTime(content)} 分钟</span></div>
          <a className="article-original-source" href={post.sourceUrl} target="_blank" rel="noreferrer" aria-label={`微信原文：${post.title}`}>微信原文：<span className="original-source-link">查看原文</span><ArrowUpRight size={14} /></a>
        </header>
        <div id="nice" className="markdown-body"><ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath, remarkDirective, remarkMdniceBlocks, remarkUnswallowMath, remarkArticleHeadingIds]}
          rehypePlugins={[rehypeKatex, rehypeYamabuki]}
          components={{
            h2: ({ children, ...props }) => <><h2 {...props}>{children}</h2>{props.id === tocHeadingId && <ArticleToc items={toc} />}</>,
            img: ({ src, alt, ...props }) => <img src={src} alt={alt ?? ''} loading="lazy" decoding="async" {...props} />,
          }}
        >{markdown}</ReactMarkdown></div>
        <div className="keyword-list">{post.keywords.map((keyword) => <span key={keyword}>#{keyword}</span>)}</div>
        <section className="comments-section">
          <h2><MessageCircle size={18} /> 评论与讨论</h2>
          <p className="comments-hint">使用 GitHub 账号登录后即可留言，评论会公开保存在本站的 GitHub Issues 中。</p>
          <Utterances key={post.slug} title={post.title} />
        </section>
        {nextPost && <Link className="next-post" to={`/post/${nextPost.slug}`}><span>下一篇</span><strong>{nextPost.title}</strong><ArrowRight size={20} /></Link>}
      </article>
    </main>
  )
}
