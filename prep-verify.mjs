import fs from 'node:fs'
import path from 'node:path'

const data = fs.readFileSync('src/data/articles-data.ts', 'utf8')
const re = /slug:\s*"([^"]+)"[\s\S]{0,900}?sourceUrl:\s*"([^"]+)"/g
const urls = {}
let m
while ((m = re.exec(data))) urls[m[1]] = m[2]
const tRe = /slug:\s*"([^"]+)"[\s\S]{0,400}?title:\s*"((?:[^"\\]|\\.)*)"/g
const titles = {}
while ((m = tRe.exec(data))) titles[m[1]] = m[2]

const dir = 'src/content/articles'
const rows = []
for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.md'))) {
  const slug = f.replace(/\.md$/, '')
  const text = fs.readFileSync(path.join(dir, f), 'utf8')
  const cjk = (text.match(/[一-鿿]/g) || []).length
  rows.push({ slug, title: titles[slug] || slug, cjk, url: urls[slug] || '' })
}
rows.sort((a, b) => a.cjk - b.cjk)
const short = rows.filter(r => r.cjk < 150)
fs.writeFileSync('../verify-list.json', JSON.stringify(short, null, 1))
console.log(short.length, '篇待验证, cjk<150')
