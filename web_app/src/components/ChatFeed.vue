<template>
  <div class="space-y-12 pb-32">
    <div v-for="(msg, index) in messages" :key="index" class="border-b border-gray-100 pb-10">

      <div v-if="msg.role === 'user'" class="flex flex-col gap-4">
        <span class="font-bold text-gray-400 uppercase tracking-wider text-xs">
          {{ lang === 'Vi' ? 'Hình ảnh đầu vào' : 'Input Image' }}
        </span>
        <img v-if="msg.image" :src="msg.image" class="w-64 h-64 object-cover rounded-md shadow-sm border border-gray-200" />
      </div>

      <div v-else class="flex flex-col gap-5 mt-6">
        <span class="font-bold text-blue-600 uppercase tracking-wider text-xs">
          {{ lang === 'Vi' ? 'Phân tích từ AI' : 'AI Analysis' }}
        </span>

        <div v-if="msg.confidence" class="flex flex-wrap gap-6 text-sm bg-gray-50 p-4 rounded-md border border-gray-200">
          <p><span class="font-semibold text-gray-700">{{ lang === 'Vi' ? 'Chẩn đoán' : 'Diagnosis' }}:</span> <span class="text-gray-900">{{ msg.disease }}</span></p>
          <p><span class="font-semibold text-gray-700">{{ lang === 'Vi' ? 'Độ tin cậy' : 'Confidence' }}:</span> <span class="text-green-600 font-bold">{{ msg.confidence }}%</span></p>
        </div>

        <div class="prose prose-blue max-w-none text-gray-800 leading-relaxed" v-html="renderMarkdown(msg.text)"></div>
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