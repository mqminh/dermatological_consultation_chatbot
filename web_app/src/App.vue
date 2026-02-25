<template>
  <div class="min-h-screen bg-white text-gray-900 font-sans">
    <div class="max-w-4xl mx-auto px-6 relative">

      <Header v-model="language" />

      <div v-if="messages.length === 0" class="mt-32 text-center">
        <p class="text-gray-400 text-lg">
          {{ language === 'Vi' ? 'Hệ thống đã sẵn sàng. Vui lòng cung cấp hình ảnh để bắt đầu.' : 'System ready. Please provide an image to begin.' }}
        </p>
      </div>

      <ChatFeed :messages="messages" :lang="language" />

      <div v-if="isLoading" class="py-10 flex items-center gap-3 text-blue-600 font-medium animate-pulse">
        <svg class="animate-spin h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        {{ language === 'Vi' ? 'Đang phân tích hình ảnh và trích xuất dữ liệu y khoa...' : 'Analyzing image and extracting medical data...' }}
      </div>

    </div>

    <InputArea :isLoading="isLoading" :lang="language" @send="handleSendMessage" />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Header from './components/Header.vue'
import ChatFeed from './components/ChatFeed.vue'
import InputArea from './components/InputArea.vue'
import { sendConsultationRequest } from './services/api'

const language = ref('En')
const messages = ref([])
const isLoading = ref(false)

const handleSendMessage = async ({ file, preview }) => {
  messages.value.push({
    role: 'user',
    image: preview
  })

  isLoading.value = true
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })

  try {
    const data = await sendConsultationRequest(file, language.value)

    if (data.success) {
      messages.value.push({
        role: 'bot',
        text: data.data.consultation,
        disease: data.data.disease,
        confidence: data.data.confidence
      })
    } else {
      messages.value.push({
        role: 'bot',
        text: `**Error:** ${data.message}`
      })
    }
  } catch (error) {
    messages.value.push({
      role: 'bot',
      text: `**Connection Error:** Cannot communicate with the server.`
    })
  } finally {
    isLoading.value = false
    setTimeout(() => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
    }, 100)
  }
}
</script>