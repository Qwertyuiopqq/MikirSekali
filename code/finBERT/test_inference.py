import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


model_path = "model_finetuned"
print("Memuat model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()   # mode inference

def prediksi_sentimen(teks):
    inputs = tokenizer(
        teks,
        return_tensors="pt",
        padding="max_length", 
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilitas = F.softmax(outputs.logits, dim=-1)
    prediksi_kelas = torch.argmax(probabilitas, dim=-1).item()

    mapping = {0: "Positif", 1: "Netral", 2: "Negatif"}

    # Skor kontinu untuk XGBoost: P(positif)*1 + P(netral)*0 + P(negatif)*(-1)
    skor_kontinu = (probabilitas[0][0].item() * 1) + (probabilitas[0][2].item() * -1)

    return mapping[prediksi_kelas], skor_kontinu


if __name__ == "__main__":
    berita = [
        # === SANGAT PENDEK (1-3 kata) ===
        "Bagus sekali.",
        "Buruk.",
        "Biasa saja.",

        # === PENDEK (4-8 kata) ===
        "Laba perusahaan naik tajam.",
        "Saham turun drastis hari ini.",
        "Pasar sedang sepi pembeli.",

        # === SEDANG (10-15 kata) ===
        "Laba bersih PT Blue Bird Tbk melonjak 40% saat Lebaran.",
        "Performa saham sektor transportasi masih stagnan hari ini.",
        "Beban operasional Garuda membengkak akibat harga BBM global.",

        # === PANJANG (20+ kata) ===
        "Laba bersih PT Blue Bird Tbk melonjak 40% berkat tingginya mobilitas masyarakat saat Lebaran kemarin, kata manajemen dalam laporan resmi kuartal kedua.",
        "Beban operasional Garuda Indonesia membengkak akibat kenaikan harga bahan bakar global yang tidak diimbangi dengan kenaikan tarif penumpang, sehingga margin laba tertekan.",

        # === SANGAT PANJANG (50+ kata) ===
        "PT Blue Bird Tbk melaporkan laba bersih yang melonjak 40% pada kuartal kedua tahun ini, didorong oleh tingginya mobilitas masyarakat selama periode Lebaran serta strategi efisiensi armada yang berhasil menekan biaya operasional, kata direktur utama dalam konferensi pers di Jakarta.",
    ]

    for teks in berita:
        n_kata = len(teks.split())
        label, skor = prediksi_sentimen(teks)
        print(f"[{n_kata:>3} kata] {label:<8} | skor={skor:+.3f} | {teks[:70]}")