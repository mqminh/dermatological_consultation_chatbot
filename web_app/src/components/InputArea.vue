<template>
  <div class="fixed bottom-0 left-0 right-0 border-t border-slate-200/70 bg-white/90 px-4 py-4 shadow-[0_-12px_40px_rgba(15,23,42,0.08)] backdrop-blur-xl sm:px-6 z-20">
    <div class="mx-auto flex max-w-5xl items-end gap-3">
      <input type="file" ref="fileInput" @change="handleFileChange" accept="image/*" class="hidden" />

      <button @click="triggerFileInput" class="flex h-[52px] w-[52px] shrink-0 items-center justify-center rounded-2xl bg-slate-100 text-slate-700 transition hover:bg-sky-100 hover:text-sky-700" title="Upload Image">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      </button>

      <div v-if="selectedImage" class="flex h-[52px] flex-1 items-center justify-between rounded-2xl border border-sky-200 bg-sky-50/80 px-3">
        <div class="flex items-center gap-3 overflow-hidden">
          <img :src="selectedImage" class="h-8 w-8 shrink-0 rounded-lg object-cover shadow-sm" />
          <span class="truncate text-sm font-medium text-sky-900">{{ selectedFile.name }}</span>
        </div>
        <button @click="clearSelection" class="shrink-0 px-2 text-lg font-semibold text-sky-500 transition hover:text-rose-500">✕</button>
      </div>

      <input
          v-else
          v-model="textMessage"
          type="text"
          :placeholder="hasDiagnosis ? (lang === 'Vi' ? 'Hỏi thêm về tình trạng của bạn...' : 'Ask follow-up questions...') : (lang === 'Vi' ? 'Vui lòng tải ảnh lên trước...' : 'Please upload an image first...')"
          :disabled="!hasDiagnosis"
          @keyup.enter="handleSendText"
          class="h-[52px] flex-1 rounded-2xl border border-slate-200 bg-white px-4 text-sm outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200 disabled:cursor-not-allowed disabled:bg-slate-50"
      />

      <button
          v-if="selectedFile"
          @click="handleSendImage"
          :disabled="isLoading"
          class="h-[52px] shrink-0 rounded-2xl bg-slate-900 px-6 font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {{ lang === 'Vi' ? 'Phân tích' : 'Analyze' }}
      </button>

      <button
          v-else
          @click="handleSendText"
          :disabled="!hasDiagnosis || !textMessage.trim() || isLoading"
          class="h-[52px] shrink-0 rounded-2xl bg-gradient-to-r from-sky-500 to-cyan-500 px-6 font-semibold text-white shadow-lg shadow-sky-200 transition hover:from-sky-600 hover:to-cyan-600 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {{ lang === 'Vi' ? 'Gửi' : 'Send' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  isLoading: Boolean,
  lang: String,
  hasDiagnosis: Boolean
})

const emit = defineEmits(['send-image', 'send-text'])

const fileInput = ref(null)
const selectedFile = ref(null)
const selectedImage = ref(null)
const textMessage = ref('')

const triggerFileInput = () => {
  fileInput.value.click()
}

const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file && file.type.startsWith('image/')) {
    selectedFile.value = file
    selectedImage.value = URL.createObjectURL(file)
    textMessage.value = ''
  }
}

const clearSelection = () => {
  selectedFile.value = null
  selectedImage.value = null
  if (fileInput.value) fileInput.value.value = ''
}

const handleSendImage = () => {
  if (!selectedFile.value) return
  emit('send-image', { file: selectedFile.value, preview: selectedImage.value })
  clearSelection()
}

const handleSendText = () => {
  if (!props.hasDiagnosis || !textMessage.value.trim() || props.isLoading) return
  emit('send-text', textMessage.value.trim())
  textMessage.value = ''
}
</script>