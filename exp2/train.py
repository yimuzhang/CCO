import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)

class GradientAttributionAnalyzer:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        """
        初始化梯度归因分析器
        
        参数:
        - model_name: 预训练模型名称
        - num_labels: 分类标签数量
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = '/data1/ymzhang/bert/iter_10000/'  
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # self.model = BertForSequenceClassification.from_pretrained(
        #     model_name, 
        #     num_labels=num_labels
        # ).to(self.device)
        self.model.train()  # 设置为评估模式
        
    def create_synthetic_data(self, num_samples=10):
        """
        创建仿真文本数据
        
        参数:
        - num_samples: 样本数量
        
        返回:
        - texts: 文本列表
        - labels: 对应标签
        """
        # 仿真情感分析数据
        positive_patterns = [
            "I hate this DVD, it's awful.",
            "These are my favorite shoes of all time.  I take these shoes everywhere and would highly recommend them!",
            "I love this product, it's fantastic.",
            "Excellent quality and great service.",
            "Outstanding performance in every aspect.",
            "Highly recommended for everyone."
        ]
        
        negative_patterns = [
            "I ordered these shoes to replace the previous pair that had ripped apart at the sole. Clark said they did not make this shoe anymore, so I had to order them through Amazon. Sadly, the second pair ripped in the same place within 3 months. My son has since bought a similar pair from Sperry and they seem to be holding up much better.",
            "I hate this product, it's awful.",
            "Poor quality and bad service.",
            "Very disappointing performance.",
            "Not recommended at all."
        ]
        
        texts = []
        labels = []
        
        for i in range(num_samples):
            if i % 2 == 0:
                text = positive_patterns[i % len(positive_patterns)]
                label = 1  # 正面
            else:
                text = negative_patterns[i % len(negative_patterns)]
                label = 0  # 负面
            texts.append(text)
            labels.append(label)
            
        return texts, labels
    
    def compute_saliency(self, text, target_label=None):
        """
        计算Saliency Map归因分数
        
        参数:
        - text: 输入文本
        - target_label: 目标标签，如果为None则使用模型预测的标签
        
        返回:
        - tokens: token列表
        - saliency_scores: 归因分数
        - predicted_label: 预测标签
        """
        # 编码文本
        with torch.enable_grad():
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 获取输入嵌入
            embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
            embeddings.requires_grad_(True)
            embeddings.retain_grad()
            
            # 前向传播
            outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids', None)
            )
            
            # 获取预测结果
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(logits, dim=-1).item()
            
            # 确定目标标签
            if target_label is None:
                target_label = predicted_label
            
            # 计算目标类别的得分梯度
            target_score = logits[0, target_label]
            target_score.backward()
            
            # 计算saliency分数（梯度的L2范数）
            gradients = embeddings.grad
            print(gradients,gradients.shape)
            if gradients is not None:
                saliency_scores = torch.norm(gradients, p=2, dim=-1).squeeze().cpu().detach().numpy()
            else:
                saliency_scores = np.zeros(inputs['input_ids'].shape[1])
            
            # 解码tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().cpu().numpy())
        
        return tokens, saliency_scores, predicted_label, probs[0].detach().cpu().numpy()
    
    def visualize_attribution(self, text, tokens, saliency_scores, predicted_label, probs):
        """
        可视化归因结果
        
        参数:
        - text: 原始文本
        - tokens: token列表
        - saliency_scores: 归因分数
        - predicted_label: 预测标签
        - probs: 概率分布
        """
        # 过滤特殊token
        valid_indices = [i for i, token in enumerate(tokens) 
                        if token not in ['[CLS]', '[SEP]', '[PAD]']]
        valid_tokens = [tokens[i] for i in valid_indices]
        valid_scores = [saliency_scores[i] for i in valid_indices]
        
        # 归一化分数用于可视化
        if max(valid_scores) > 0:
            normalized_scores = [score / max(valid_scores) for score in valid_scores]
        else:
            normalized_scores = valid_scores
        
        # 创建可视化
        plt.figure(figsize=(12, 6))
        
        # 绘制热力图
        short=['i',',','.','!','of','all','are']
        long=['favorite','everywhere','recommend']
       # plt.subplot(2, 1, 1)
        colors = plt.cm.Reds(normalized_scores)
        pos=0
        # current_pos = pos  # 起始x坐标

        # for i, (token, score) in enumerate(zip(valid_tokens, normalized_scores)):
        #     text_obj = plt.text(current_pos, 0, token, ha='center', va='center', 
        #                         fontsize=11, bbox=dict(boxstyle="round,pad=0.2", 
        #                                             facecolor=colors[i], 
        #                                             alpha=0.7))
            
        #     plt.gcf().canvas.draw()
        #     bbox = text_obj.get_bbox_patch()
        #     bbox_width = bbox.get_width()  # 获取框的宽度（单位：点或像素）
            
        #     # 将宽度转换为数据坐标（可选：根据实际坐标系调整）
        #     # 这里假设x轴为数据坐标，y轴固定为0
        #     inv = plt.gca().transData.inverted()
        #     bbox_data_width = inv.transform((bbox_width, 0))[0] - inv.transform((0, 0))[0]
            
        #     # 更新下一个框的位置：当前框位置 + 当前框宽度
        #     current_pos += bbox_data_width*30
        l0=[0.11,0.21,0.35,0.44,0.53,0.68,0.79,0.8,0.82,0.87]
        for i, (token, score) in enumerate(zip(valid_tokens, normalized_scores)):
            l=len(token)
            tmp=l0[l-1]
            text_obj=plt.text(pos+tmp, 0, token, ha='center', va='center', 
                    fontsize=11, bbox=dict(boxstyle="round,pad=0.2", 
                                        facecolor=colors[i], 
                                        alpha=0.7))
            pos=pos+tmp*2
            print(text_obj.get_window_extent(renderer=None).size)
        plt.xlim(-1, len(valid_tokens))
        plt.ylim(-1, 1)
        plt.axis('off')
        
        # # 绘制分数条形图
        # plt.subplot(2, 1, 2)
        # x_pos = np.arange(len(valid_tokens))
        # plt.bar(x_pos, valid_scores, alpha=0.7, color='skyblue')
        # plt.xlabel('Token Position')
        # plt.ylabel('Saliency Score')
        # plt.title('Token Importance Scores')
        # plt.xticks(x_pos, valid_tokens, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def analyze_text(self, text, target_label=None):
        """
        完整的文本分析流程
        
        参数:
        - text: 输入文本
        - target_label: 目标标签
        """
        print(f"Analyzing text: {text}")
        
        # 计算归因分数
        tokens, saliency_scores, predicted_label, probs = self.compute_saliency(
            text, target_label
        )
        
        # 打印结果
        print(f"Predicted label: {predicted_label}")
        print(f"Class probabilities: {probs}")
        print("\nToken importance scores:")
        for i, (token, score) in enumerate(zip(tokens, saliency_scores)):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                print(f"{token:15s}: {score:.6f}")
        
        # 可视化
        print(saliency_scores)
        self.visualize_attribution(text, tokens, saliency_scores, predicted_label, probs)
        
        return saliency_scores

# 使用示例
def main():
    # 初始化分析器
    analyzer = GradientAttributionAnalyzer()
    
    # 创建仿真数据
    texts, labels = analyzer.create_synthetic_data(num_samples=5)
    
    # 分析每个文本
    for i, (text, true_label) in enumerate(zip(texts, labels)):
        print(f"\n{'='*50}")
        print(f"Sample {i+1}: True label = {true_label}")
        print(f"{'='*50}")
        
        saliency_scores = analyzer.analyze_text(text)

        print(f"\nMax importance token: {analyzer.tokenizer.convert_ids_to_tokens([np.argmax(saliency_scores)])[0]}")
        print(f"Max importance score: {np.max(saliency_scores):.6f}")

if __name__ == "__main__":
    main()