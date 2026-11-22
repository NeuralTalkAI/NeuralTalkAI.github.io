# -*- coding: utf-8 -*-
"""构建后处理：为每篇文章生成 SEO 静态入口页 dist/post/<slug>/index.html，
生成 404.html、.nojekyll，并用无 # 的真实 URL 重写 sitemap/rss/robots。"""
import re, json, html, os, io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(APP, 'dist')
BASE = 'https://neuraltalk.cc'

src = open(os.path.join(APP, 'src', 'data', 'articles-data.ts'), encoding='utf-8').read()
STR = r'(?:[^"\\]|\\.)*'
posts = re.findall(
    r'\{\s*slug: "([^"]+)",\s*title: "(' + STR + r')",\s*date: "([^"]+)"[\s\S]*?excerpt: "(' + STR + r')",[\s\S]*?sourceUrl: "([^"]+)"',
    src)
# 加上手写的 monomoe 示例
posts.insert(0, ('monomoe-weight-major-megakernel', 'MonoMoE：权重主导融合 MegaKernel，量化 MoE 解码提速最高 1.54 倍',
                 '2026-11-09', '亚马逊 AGI 团队提出权重主导的持久化 MegaKernel MonoMoE，在 H200 上较 vLLM 基线提速最高 1.54 倍，端到端单 Token 延迟最高降低 18.7%。',
                 'https://mp.weixin.qq.com/s/EC8Ush6dzqP_ODbLoqOdcg'))
print('posts:', len(posts))

template = open(os.path.join(DIST, 'index.html'), encoding='utf-8').read()

def esc(s):
    return html.escape(s, quote=True)

def make_page(title, desc, canonical):
    head = (f'<title>{esc(title)}</title>\n'
            f'    <meta name="description" content="{esc(desc)}" />\n'
            f'    <link rel="canonical" href="{esc(canonical)}" />\n'
            f'    <meta property="og:type" content="article" />\n'
            f'    <meta property="og:title" content="{esc(title)}" />\n'
            f'    <meta property="og:description" content="{esc(desc)}" />\n'
            f'    <meta property="og:url" content="{esc(canonical)}" />\n'
            f'    <meta property="og:site_name" content="NeuralTalk" />')
    page = template.replace('<title>NeuralTalk</title>', head, 1)
    # 避免与模板自带 description 重复，删掉旧的
    page = page.replace('<meta name="description" content="NeuralTalk，记录 AI、计算系统与工程实践。" />', '', 1)
    return page

n = 0
for slug, title, date, excerpt, srcurl in posts:
    t = json.loads('"' + title + '"')
    e = json.loads('"' + excerpt + '"')
    d = os.path.join(DIST, 'post', slug)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'index.html'), 'w', encoding='utf-8').write(
        make_page(f'{t} | NeuralTalk', e, f'{BASE}/post/{slug}'))
    n += 1
print('post pages:', n)

# 归档、关于、首页
for path, ttl in [('archive', f'归档 | NeuralTalk'), ('about', f'关于 | NeuralTalk')]:
    d = os.path.join(DIST, path)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'index.html'), 'w', encoding='utf-8').write(
        make_page(ttl, 'NeuralTalk，记录 AI、计算系统与工程实践。', f'{BASE}/{path}/'))

# 404 兜底（SPA 回退）
open(os.path.join(DIST, '404.html'), 'w', encoding='utf-8').write(template)
open(os.path.join(DIST, '.nojekyll'), 'w').close()

# sitemap / rss / robots（真实 URL）
sm = ['<?xml version="1.0" encoding="UTF-8"?>',
      '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
      f'<url><loc>{BASE}/</loc></url>',
      f'<url><loc>{BASE}/archive/</loc></url>',
      f'<url><loc>{BASE}/about/</loc></url>']
rss = ['<?xml version="1.0" encoding="UTF-8"?>', '<rss version="2.0"><channel>',
       '<title>NeuralTalk</title>', f'<link>{BASE}/</link>',
       '<description>记录 AI、计算系统与工程实践</description>', '<language>zh-CN</language>']
for slug, title, date, excerpt, srcurl in posts:
    t = json.loads('"' + title + '"')
    e = json.loads('"' + excerpt + '"')
    sm.append(f'<url><loc>{BASE}/post/{slug}/</loc><lastmod>{date}</lastmod></url>')
    rss.append(f'<item><title>{esc(t)}</title><link>{BASE}/post/{slug}/</link>'
               f'<guid>{BASE}/post/{slug}/</guid><pubDate>{date}</pubDate>'
               f'<description>{esc(e)}</description></item>')
sm.append('</urlset>')
rss.append('</channel></rss>')
open(os.path.join(DIST, 'sitemap.xml'), 'w', encoding='utf-8').write('\n'.join(sm))
open(os.path.join(DIST, 'rss.xml'), 'w', encoding='utf-8').write('\n'.join(rss))
open(os.path.join(DIST, 'robots.txt'), 'w', encoding='utf-8').write(
    f'User-agent: *\nAllow: /\n\nSitemap: {BASE}/sitemap.xml\n')
print('sitemap/rss/robots/404/.nojekyll written')
