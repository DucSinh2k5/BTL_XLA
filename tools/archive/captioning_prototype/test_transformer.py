from transformers import pipeline

# Có thể thay bằng "Salesforce/blip-image-captioning-base" để chất lượng tốt hơn
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

result = captioner(r"F:\Downloads\sun_aaalbzqrimafwbiv.jpg")
print(result[0]["generated_text"])