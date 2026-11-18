import monomoe from '../content/posts/monomoe.md?raw'
import { posts as importedPosts, type PostMeta } from './articles-data'

export type Post = PostMeta & { content?: string }

const monomoePost: Post = {
  slug: 'monomoe-weight-major-megakernel',
  title: 'MonoMoE：权重主导融合 MegaKernel，量化 MoE 解码提速最高 1.54 倍',
  date: '2026-11-09',
  original: true,
  category: '机器学习系统',
  keywords: ['MoE 解码', '权重主导调度', '持久化 MegaKernel', 'FP8 量化', 'vLLM'],
  excerpt: '亚马逊 AGI 团队提出权重主导的持久化 MegaKernel MonoMoE，在 H200 上较 vLLM 基线提速最高 1.54 倍，端到端单 Token 延迟最高降低 18.7%。',
  sourceUrl: 'https://mp.weixin.qq.com/s/EC8Ush6dzqP_ODbLoqOdcg',
  content: monomoe,
}

/** Newest first: the monomoe sample (2026-11-09) predates the Excel snapshot, so it stays on top. */
export const posts: Post[] = [monomoePost, ...importedPosts]

const categoryCounts = new Map<string, number>()
for (const post of posts) categoryCounts.set(post.category, (categoryCounts.get(post.category) ?? 0) + 1)
export const categories = ['全部', ...[...categoryCounts.entries()]
  .sort((a, b) => b[1] - a[1])
  .filter(([, count]) => count >= 3)
  .slice(0, 10)
  .map(([name]) => name)]

/** 关键词标签索引：按出现篇数排序，供侧边栏展示与筛选。 */
const tagCounts = new Map<string, number>()
for (const post of posts) {
  for (const keyword of post.keywords) {
    const trimmed = keyword.trim()
    if (trimmed) tagCounts.set(trimmed, (tagCounts.get(trimmed) ?? 0) + 1)
  }
}
export const tagIndex: [string, number][] = [...tagCounts.entries()]
  .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'zh-CN'))
  .slice(0, 24)

const loaders = import.meta.glob('../content/articles/*.md', {
  query: '?raw',
  import: 'default',
  eager: false,
}) as Record<string, () => Promise<string>>

export async function loadPostContent(slug: string): Promise<string | undefined> {
  const loader = loaders[`../content/articles/${slug}.md`]
  if (loader) return loader()
  return posts.find((post) => post.slug === slug)?.content
}
