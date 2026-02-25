<template>
  <header class="flex justify-between items-center py-6 border-b border-gray-200 mb-8 bg-white sticky top-0 z-10">
    <h1 class="text-2xl font-bold text-gray-900 tracking-tight">Dermatology AI</h1>
    <select
        :value="modelValue"
        @change="$emit('update:modelValue', $event.target.value)"
        class="border border-gray-300 rounded-md px-4 py-2 bg-white text-sm font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
    >
      <option value="En">English</option>
      <option value="Vi">Tiếng Việt</option>
    </select>
  </header>
</template>

<script setup>
defineProps({
  modelValue: String
})
defineEmits(['update:modelValue'])
</script>