<template>
  <div class="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 p-4 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
    <div class="max-w-4xl mx-auto flex items-end gap-3">

      <input type="file" ref="fileInput" @change="handleFileChange" accept="image/*" class="hidden" />

      <button @click="triggerFileInput" class="p-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors shrink-0 flex items-center justify-center" title="Upload Image">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      </button>

      <div v-if="selectedImage" class="flex-1 border border-blue-200 rounded-md p-2 flex items-center justify-between bg-blue-50 h-[52px]">
        <div class="flex items-center gap-3 overflow-hidden">
          <img :src="selectedImage" class="w-8 h-8 object-cover rounded shadow-sm shrink-0" />
          <span class="text-sm font-medium text-blue-900 truncate">{{ selectedFile.name }}</span>
        </div>
        <button @click="clearSelection" class="text-blue-400 hover:text-red-600 px-2 font-bold shrink-0">✕</button>
      </div>

      <input
          v-else
          v-model="textMessage"
          type="text"
          :placeholder="hasDiagnosis ? (lang === 'Vi' ? 'Hỏi thêm về tình trạng của bạn...' : 'Ask follow-up questions...') : (lang === 'Vi' ? 'Vui lòng tải ảnh lên trước...' : 'Please upload an image first...')"
          :disabled="!hasDiagnosis"
          @keyup.enter="handleSendText"
          class="flex-1 h-[52px] px-4 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:cursor-not-allowed"
      />

      <button
          v-if="selectedFile"
          @click="handleSendImage"
          :disabled="isLoading"
          class="px-6 h-[52px] bg-gray-900 text-white font-semibold rounded-md hover:bg-black disabled:opacity-50 transition-colors shrink-0"
      >
        {{ lang === 'Vi' ? 'Phân tích' : 'Analyze' }}
      </button>

      <button
          v-else
          @click="handleSendText"
          :disabled="!hasDiagnosis || !textMessage.trim() || isLoading"
          class="px-6 h-[52px] bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0"
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