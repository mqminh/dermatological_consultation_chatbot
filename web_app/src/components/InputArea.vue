<template>
  <div class="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 p-4 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
    <div class="max-w-4xl mx-auto flex items-end gap-4">
      <input type="file" ref="fileInput" @change="handleFileChange" accept="image/*" class="hidden" />

      <button @click="triggerFileInput" class="p-4 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors shrink-0">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      </button>

      <div v-if="selectedImage" class="flex-1 border border-blue-200 rounded-md p-2 flex items-center justify-between bg-blue-50">
        <div class="flex items-center gap-4">
          <img :src="selectedImage" class="w-12 h-12 object-cover rounded shadow-sm" />
          <span class="text-sm font-medium text-blue-900 truncate">{{ selectedFile.name }}</span>
        </div>
        <button @click="clearSelection" class="text-blue-400 hover:text-red-600 px-3 py-1 font-bold text-lg">✕</button>
      </div>

      <div v-else class="flex-1 p-4 border-2 border-dashed border-gray-300 rounded-md text-gray-400 text-sm flex items-center justify-center bg-gray-50 cursor-pointer hover:bg-gray-100 transition-colors" @click="triggerFileInput">
        {{ lang === 'Vi' ? 'Nhấn để chọn hình ảnh cần phân tích...' : 'Click to select an image for analysis...' }}
      </div>

      <button @click="handleSend" :disabled="!selectedFile || isLoading" class="px-8 py-4 bg-gray-900 text-white font-semibold rounded-md hover:bg-black disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0">
        {{ lang === 'Vi' ? 'Gửi' : 'Analyze' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  isLoading: Boolean,
  lang: String
})

const emit = defineEmits(['send'])

const fileInput = ref(null)
const selectedFile = ref(null)
const selectedImage = ref(null)

const triggerFileInput = () => {
  fileInput.value.click()
}

const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file && file.type.startsWith('image/')) {
    selectedFile.value = file
    selectedImage.value = URL.createObjectURL(file)
  }
}

const clearSelection = () => {
  selectedFile.value = null
  selectedImage.value = null
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const handleSend = () => {
  if (!selectedFile.value) return
  emit('send', { file: selectedFile.value, preview: selectedImage.value })
  clearSelection()
}
</script>