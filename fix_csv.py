import csv
p = "D:/Personal_work/杂/作业/人工智能/rubbish(deepseek)/mobilenet_project_v4/result/summary.csv"
rows = [
    ['filename', 'prediction', 'confidence', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
    ['cardboard_test.jpg', 'cardboard', '0.6308', '0.6308', '0.0010', '0.0021', '0.3302', '0.0007', '0.0353'],
    ['glass_test.jpg', 'glass', '0.5817', '0.0192', '0.5817', '0.2349', '0.0039', '0.0325', '0.1278'],
    ['metal_test.jpg', 'trash', '0.8909', '0.0006', '0.0066', '0.0335', '0.0020', '0.0665', '0.8909'],
    ['paper_test.jpg', 'paper', '0.9926', '0.0017', '0.0019', '0.0004', '0.9926', '0.0001', '0.0034'],
    ['plastic_test.jpg', 'plastic', '0.5039', '0.0006', '0.0373', '0.1990', '0.0029', '0.5039', '0.2564'],
    ['test.jpg', 'cardboard', '0.9865', '0.9865', '0.0000', '0.0013', '0.0026', '0.0000', '0.0097'],
    ['trash_test.jpg', 'trash', '0.4007', '0.0221', '0.0239', '0.1787', '0.2659', '0.1087', '0.4007'],
]
with open(p, 'w', encoding='utf-8-sig', newline='') as f:
    csv.writer(f).writerows(rows)
print('done')
