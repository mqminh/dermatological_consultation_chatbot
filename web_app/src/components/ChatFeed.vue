<template>
  <div class="space-y-6 pb-32">
    <div v-for="(msg, index) in messages" :key="index" class="rounded-[24px] border border-slate-100 bg-white/90 p-4 shadow-sm sm:p-5">
      <div v-if="msg.role === 'user'" class="flex flex-col gap-4">
        <span class="text-xs font-semibold uppercase tracking-[0.25em] text-slate-400">
          {{ lang === 'Vi' ? 'Bạn' : 'You' }}
        </span>
        <img v-if="msg.image" :src="msg.image" class="h-64 w-64 rounded-2xl border border-slate-200 object-cover shadow-sm" />
        <p v-if="msg.text" class="max-w-2xl rounded-2xl bg-slate-900 px-4 py-3 text-sm text-white shadow-sm">{{ msg.text }}</p>
      </div>

      <div v-else class="flex flex-col gap-4">
        <span class="text-xs font-semibold uppercase tracking-[0.25em] text-sky-600">
          {{ lang === 'Vi' ? 'Phân tích từ AI' : 'AI Analysis' }}
        </span>

        <div v-if="msg.confidence" class="flex flex-wrap gap-4 rounded-2xl border border-sky-100 bg-sky-50/70 p-4 text-sm text-slate-700">
          <p><span class="font-semibold text-slate-700">{{ lang === 'Vi' ? 'Chẩn đoán' : 'Diagnosis' }}:</span> <span class="ml-1 text-slate-900">{{ msg.disease }}</span></p>
          <p><span class="font-semibold text-slate-700">{{ lang === 'Vi' ? 'Độ tin cậy' : 'Confidence' }}:</span> <span class="ml-1 font-semibold text-emerald-600">{{ msg.confidence }}%</span></p>
        </div>

        <div class="markdown-body rounded-2xl bg-slate-50/80 p-4 text-sm leading-7 text-slate-700" v-html="renderMarkdown(msg.text)"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { renderMarkdown } from '../utils/markdown'

defineProps({
  messages: Array,
  lang: String
})
</script>