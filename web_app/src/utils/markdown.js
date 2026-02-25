import { marked } from 'marked'

export const renderMarkdown = (text) => {
    if (!text) return ''
    return marked(text)
}