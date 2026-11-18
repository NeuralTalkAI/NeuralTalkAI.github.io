import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, NavLink, Route, Routes, useParams } from 'react-router'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkDirective from 'remark-directive'
import rehypeKatex from 'rehype-katex'
import remarkMdniceBlocks, { normalizeMdniceContainers } from './markdown/remarkMdniceBlocks'
import rehypeYamabuki from './markdown/rehypeYamabuki'
import { getArticleToc, normalizeArticleImageUrls, remarkArticleHeadingIds, stripManualTocList, type TocItem } from './markdown/articleStructure'
import 'katex/dist/katex.min.css'
import { ArrowLeft, ArrowRight, ArrowUpRight, MessageCircle, Search, X } from 'lucide-react'
import { categories, loadPostContent, posts, tagIndex } from './data/posts'
import './App.css'
import './markdown/yamabuki.css'

const siteName = 'NeuralTalk'
const brandMark = 'NT'
const authorName = 'NeuralTalk'
const authorMark = 'NT'

function formatDate(date: string) {
  return new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' }).format(new Date(`${date}T12:00:00`))
}

function getReadingTime(content: string) {
  const plainText = content
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/\$[^$\n]+\$/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[#>*_`~|]/g, ' ')
  const chineseCharacters = plainText.match(/[一-鿿]/g)?.length ?? 0
  const englishWords = plainText.match(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g)?.length ?? 0
  return Math.max(1, Math.ceil(chineseCharacters / 400 + englishWords / 180))
}

function Header({ onSearch }: { onSearch: () => void }) {
  return (
    <header className="site-header">
      <div className="header-inner">
        <Link to="/" className="brand" aria-label={`${siteName}首页`}>
          <span className="brand-mark">{brandMark}</span>
          <span>{siteName}</span>
        </Link>
        <nav className="main-nav" aria-label="主导航">
          <NavLink to="/" end>首页</NavLink>
          <NavLink to="/archive">归档</NavLink>
          <NavLink to="/about">关于</NavLink>
          <button className="icon-button" onClick={onSearch} aria-label="搜索文章"><Search size={19} /></button>
        </nav>
      </div>
    </header>
  )
}

function Sidebar({ activeTag, onSelectTag }: { activeTag: string, onSelectTag: (tag: string) => void }) {
  return (
    <aside className="sidebar">
      <div className="profile-avatar">{authorMark}</div>
      <h2>{authorName}</h2>
      <p className="profile-role">教育／科研工作者</p>
      <p className="profile-bio">记录 AI、计算系统与工程实践，整理值得反复阅读的内容。</p>
      <div className="profile-stats">
        <div><strong>{posts.length.toLocaleString()}</strong><span>已收录文章</span></div>
        <div><strong>{posts.filter((post) => post.original).length.toLocaleString()}</strong><span>原创文章</span></div>
      </div>
      <div className="sidebar-tags" aria-label="按标签筛选">
        <p className="sidebar-tags-title">标签索引</p>
        <div className="sidebar-tag-list">
          {tagIndex.map(([tag, count]) => (
            <button key={tag} className={activeTag === tag ? 'active' : ''} onClick={() => onSelectTag(activeTag === tag ? '' : tag)}>
              #{tag}<span className="tag-count">{count}</span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  )
}

function PostCard({ post, index }: { post: typeof posts[number], index: number }) {
  return (
    <article className="post-card">
      <div className="post-meta">
        <time dateTime={post.date}>{formatDate(post.date)}</time>
        <span className="meta-dot">·</span>
        <span>{post.category}</span>
        {post.original && <span className="original-badge">原创</span>}
      </div>
      <h2><Link to={`/post/${post.slug}`}>{post.title}</Link></h2>
      <p>{post.excerpt}</p>
      <Link className="read-more" to={`/post/${post.slug}`}>阅读全文 <ArrowRight size={16} /></Link>
      {index === 0 && <span className="latest-label">最新</span>}
    </article>
  )
}

function Home({ tagFilter, onSelectTag }: { tagFilter: string, onSelectTag: (tag: string) => void }) {
  const [category, setCategory] = useState('全部')
  const [visible, setVisible] = useState(40)
  useEffect(() => { setVisible(40) }, [category, tagFilter])
  const filtered = posts.filter((post) =>
    (category === '全部' || post.category === category) &&
    (!tagFilter || post.keywords.some((keyword) => keyword.trim() === tagFilter)))
  const shown = filtered.slice(0, visible)
  return (
    <main className="page-shell home-layout">
      <Sidebar activeTag={tagFilter} onSelectTag={onSelectTag} />
      <section className="content-column">
        <div className="section-heading">
          <div><p className="section-kicker">KNOWLEDGE ARCHIVE</p><h1>最近发布</h1></div>
          <span className="post-count">{filtered.length} 篇文章</span>
        </div>
        <div className="category-filter" aria-label="按分类筛选">
          {categories.map((item) => <button key={item} className={category === item ? 'active' : ''} onClick={() => setCategory(item)}>{item}</button>)}
        </div>
        {tagFilter && (
          <div className="active-tag-bar">
            <span>当前标签：<strong>#{tagFilter}</strong></span>
            <button onClick={() => onSelectTag('')}><X size={14} />清除标签筛选</button>
          </div>
        )}
        <div className="post-list">
          {shown.map((post, index) => <PostCard key={post.slug} post={post} index={index} />)}
        </div>
        {filtered.length > shown.length && (
          <button className="load-more" onClick={() => setVisible((count) => count + 40)}>
            加载更多（还有 {filtered.length - shown.length} 篇）
          </button>
        )}
      </section>
    </main>
  )
}

function Archive() {
  const grouped = useMemo(() => posts.reduce<Record<string, typeof posts>>((acc, post) => {
    const year = post.date.slice(0, 4)
    acc[year] = [...(acc[year] || []), post]
    return acc
  }, {}), [])
  return (
    <main className="page-shell narrow-page">
      <div className="page-title"><p className="section-kicker">ARCHIVE</p><h1>文章归档</h1><p>按时间顺序整理，保留每一次发布记录。</p></div>
      {Object.entries(grouped).map(([year, yearPosts]) => (
        <section className="archive-year" key={year}>
          <h2>{year}<span>{yearPosts.length} 篇</span></h2>
          {yearPosts.map((post) => <Link className="archive-row" to={`/post/${post.slug}`} key={post.slug}><time>{post.date.slice(5).replace('-', ' / ')}</time><span>{post.title}</span><ArrowUpRight size={18} /></Link>)}
        </section>
      ))}
    </main>
  )
}

function About() {
  return (
    <main className="page-shell narrow-page about-page">
      <div className="page-title"><p className="section-kicker">ABOUT</p><h1>关于这个知识库</h1></div>
      <p className="about-lead">把散落在公众号里的内容，整理成一个可以搜索、引用和持续更新的个人知识库。</p>
      <div className="about-grid">
        <section><h2>这里收录什么</h2><p>主要关注 AI、计算系统、逆向工程与工程实践。文章保留发布时间、原创标记、关键词和公众号原始链接。</p></section>
        <section><h2>如何参与讨论</h2><p>文章底部已接入 Utterances。读者可以使用 GitHub 账号留言，评论会保存在仓库 Issues 中，公开且便于维护。</p></section>
        <section><h2>内容如何维护</h2><p>每篇文章对应一个 Markdown 文件，图片与文章一起放入仓库。更新内容后，GitHub Pages 会自动重新生成网站。</p></section>
      </div>
      <Link to="/" className="primary-link">开始阅读 <ArrowRight size={17} /></Link>
    </main>
  )
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
    script.setAttribute('repo', 'NeuralTalkAI/neuraltalk')
    script.setAttribute('issue-term', 'title')
    script.setAttribute('theme', 'github-light')
    script.crossOrigin = 'anonymous'
    script.async = true
    container.appendChild(script)
  }, [title])
  return <div className="utterances-container" ref={containerRef} />
}

function Article() {
  const { slug } = useParams()
  const post = posts.find((item) => item.slug === slug)
  const [content, setContent] = useState<string | null>(post?.content ?? null)
  useEffect(() => { window.scrollTo(0, 0) }, [slug])
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
  if (!post) return <NotFound />
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
          remarkPlugins={[remarkGfm, remarkMath, remarkDirective, remarkMdniceBlocks, remarkArticleHeadingIds]}
          rehypePlugins={[rehypeKatex, rehypeYamabuki]}
          components={{ h2: ({ children, ...props }) => <><h2 {...props}>{children}</h2>{props.id === tocHeadingId && <ArticleToc items={toc} />}</> }}
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

function NotFound() {
  return <main className="page-shell narrow-page not-found"><p className="section-kicker">404</p><h1>没有找到这篇文章</h1><p>文章可能已移动或尚未收录。</p><Link to="/" className="primary-link"><ArrowLeft size={17} />返回首页</Link></main>
}

function SearchDialog({ open, onClose }: { open: boolean, onClose: () => void }) {
  const [query, setQuery] = useState('')
  useEffect(() => {
    if (!open) setQuery('')
    const handleKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [open, onClose])
  if (!open) return null
  const results = posts.filter((post) => `${post.title}${post.excerpt}${post.keywords.join('')}`.toLowerCase().includes(query.toLowerCase()))
  return (
    <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <section className="search-dialog" role="dialog" aria-modal="true" aria-label="搜索文章">
        <div className="search-input-wrap"><Search size={21} /><input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索标题、摘要或关键词…" /><button className="icon-button" onClick={onClose} aria-label="关闭搜索"><X size={20} /></button></div>
        <p className="search-result-count">{query ? `找到 ${results.length} 篇相关文章` : '输入关键词开始搜索'}</p>
        <div className="search-results">{query && results.map((post) => <Link key={post.slug} to={`/post/${post.slug}`} onClick={onClose}><span>{post.category}</span><strong>{post.title}</strong><ArrowUpRight size={18} /></Link>)}{query && results.length === 0 && <div className="empty-state">没有找到相关内容，试试其他关键词。</div>}</div>
      </section>
    </div>
  )
}

export default function App() {
  const [searchOpen, setSearchOpen] = useState(false)
  const [tagFilter, setTagFilter] = useState('')
  return (
    <div className="app">
      <Header onSearch={() => setSearchOpen(true)} />
      <Routes>
        <Route path="/" element={<Home tagFilter={tagFilter} onSelectTag={setTagFilter} />} />
        <Route path="/archive" element={<Archive />} />
        <Route path="/about" element={<About />} />
        <Route path="/post/:slug" element={<Article />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
      <footer className="site-footer"><span>© 2026 {siteName}</span><span>Markdown · GitHub Pages · Utterances</span></footer>
      <SearchDialog open={searchOpen} onClose={() => setSearchOpen(false)} />
    </div>
  )
}
