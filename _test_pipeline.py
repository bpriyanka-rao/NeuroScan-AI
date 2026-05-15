import sys
sys.path.insert(0, 'src')
from predict import predict_image
from gradcam import generate_gradcam
from clinical_insights import get_clinical_insights
from pdf_report import generate_pdf_report

result = predict_image('dummy.jpg')
b64 = generate_gradcam('dummy.jpg', model=None)
ins = get_clinical_insights(result['class'], result['confidence'])
pdf = generate_pdf_report(result, ins, patient_name='Test Patient')
print('ALL MODULES OK')
print('  predict:', result['class'], str(result['confidence']) + '%')
print('  gradcam len:', len(b64))
print('  insights stage:', ins['stage_info']['name'])
print('  pdf size:', len(pdf), 'bytes')
