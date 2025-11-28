using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TextCore.Text;
using UnityEngine.VFX;

public class BlazePoseDataFile : ScriptableObject
{
    public Actor Character; 
    private bool file_exist = false;
    private float scale = 1.0f;

    private List<string> csvFileNameList = new List<string>();
    public int selectedData = 0;

    public Dictionary<int, List<Vector3>> frameDict;

    // 기본 CSV 폴더 경로.. 그냥 절대경로로 함
    public string defaultCsvFolderPath = @"C:\Users\KDY\CAU\3-2\AIML\AI-ML-FinalProject\AI_ML_blazepose";

    private void OnEnable()
    {
        if (frameDict == null || frameDict.Count == 0)
        {
            AutoLoadLatestCSV();
        }
    }

    // defaultCsvFolderPath에서 가장 최신 CSV 자동 로드
    private void AutoLoadLatestCSV()
    {
        if (string.IsNullOrEmpty(defaultCsvFolderPath))
        {
            Debug.LogWarning("[AutoCSV] defaultCsvFolderPath 가 비어 있습니다.");
            return;
        }

        if (!Directory.Exists(defaultCsvFolderPath))
        {
            Debug.LogError($"[AutoCSV] 폴더를 찾을 수 없습니다: {defaultCsvFolderPath}");
            return;
        }

        string[] files = Directory.GetFiles(defaultCsvFolderPath, "*.csv");
        if (files.Length == 0)
        {
            Debug.LogError($"[AutoCSV] CSV 파일이 없습니다: {defaultCsvFolderPath}");
            return;
        }

        // 가장 최근에 수정된 CSV 선택
        string latestFile = null;
        System.DateTime latestTime = System.DateTime.MinValue;

        foreach (var f in files)
        {
            var info = new FileInfo(f);
            if (info.LastWriteTime > latestTime)
            {
                latestTime = info.LastWriteTime;
                latestFile = f;
            }
        }

        Debug.Log($"[AutoCSV] 최신 CSV 자동 로드: {latestFile}");
        ImportCSVDataFromPath(latestFile, scale);
    }

    public void MotionCSVFile_Inspector(Actor _actor)
    {
        Character = _actor;

        EditorGUILayout.BeginHorizontal();
        GUILayout.Space(10);
        GUI.color = Color.white;
        EditorGUILayout.HelpBox("Motion CSV Loader", MessageType.None);
        EditorGUILayout.EndHorizontal();

        EditorGUILayout.BeginHorizontal();
        // CSV 폴더 불러오기 버튼
        if (GUILayout.Button("CSV Loader: Load CSV File Directory"))
        {
            EditorApplication.delayCall += () =>
            {
                string dataPath = EditorUtility.OpenFolderPanel("CSV File Folder", "", "Assets");
                Debug.Log(dataPath);
                file_exist = LoadCSVDirectory(dataPath, "*.csv");
            };
        }
        EditorGUILayout.EndHorizontal();

        
        // CSV 파일 목록이 있으면 팝업으로 선택
        // Label과 Popup 분리 + Popup 너비 강제 지정
        EditorGUILayout.LabelField("Select CSV Data", GUILayout.Width(120));

        if (csvFileNameList.Count > 0)
        {
            EditorGUILayout.BeginHorizontal();
            int newIndex = EditorGUILayout.IntField(selectedData, GUILayout.Width(50));
            if (newIndex != selectedData)
            {
                // 입력한 값이 유효한 범위라면 적용
                if (newIndex >= 0 && newIndex < csvFileNameList.Count)
                {
                    selectedData = newIndex;
                    GUI.FocusControl(null); // 입력 포커스 해제
                }
                else
                {
                    Debug.LogWarning($"⚠️ Invalid index {newIndex}. Range is 0 to {csvFileNameList.Count - 1}.");
                }
            }

            if (selectedData >= 0 && selectedData < csvFileNameList.Count)
            {
                EditorGUILayout.LabelField("Selected File: " + csvFileNameList[selectedData], GUILayout.Width(550));
            }
            // selectedData = EditorGUILayout.Popup(selectedData, csvFileNameList.ToArray(),
            // GUILayout.ExpandWidth(true),  // ✅ 폭 자동 확장
            // GUILayout.MinWidth(150)       // 최소 폭 확보
            // );
            EditorGUILayout.EndHorizontal();

            if (GUILayout.Button("Import CSV", GUILayout.Width(150)))
            {
                ImportCSVData(selectedData, scale);
            }
            

        }
        
        
    }
    // 폴더 내 CSV 파일을 불러오는 함수
    private bool LoadCSVDirectory(string path, string extension)
    {
        csvFileNameList.Clear();

        if (string.IsNullOrEmpty(path)) return false;

        string[] files = Directory.GetFiles(path, extension);
        foreach (var f in files)
        {
            csvFileNameList.Add(f);
        }
        
        return csvFileNameList.Count > 0;
    }

// ✅ 인덱스 기반 버전: 기존 UI에서 사용하는 함수
    public void ImportCSVData(int index, float scale)
    {
        if (index < 0 || index >= csvFileNameList.Count) return;

        string filePath = csvFileNameList[index];
        Debug.Log($"[CSV Loader] Importing: {filePath}");

        ImportCSVDataFromPath(filePath, scale);
    }

    // ✅ 실제 CSV 파싱 로직: 경로만 주면 어디서든 재사용
    private void ImportCSVDataFromPath(string filePath, float scale)
    {
        if (string.IsNullOrEmpty(filePath))
        {
            Debug.LogError("[CSV Loader] filePath 가 비어 있습니다.");
            return;
        }

        if (!File.Exists(filePath))
        {
            Debug.LogError($"[CSV Loader] CSV 파일이 존재하지 않습니다: {filePath}");
            return;
        }

        Debug.Log($"[CSV Loader] Importing: {filePath}");

        try
        {
            string[] lines = File.ReadAllLines(filePath);

            // frame 단위로 landmark 데이터를 저장하는 리스트
            frameDict = new Dictionary<int, List<Vector3>>();

            int nJoint = 33;
            int nFrame = (lines.Length - 1) / nJoint; // 첫 줄은 헤더 제외
            Debug.Log($"nLines : {lines.Length} , nFrame : {nFrame}");

            // (1) 단순 프레임별 리스트 생성
            for (int f = 0; f < nFrame; f++)
            {
                List<Vector3> frame = new List<Vector3>();
                List<float> frame_quality = new List<float>();
                for (int j = 0; j < nJoint; j++)
                {
                    int lineIndex = 1 + f * nJoint + j; // 헤더 한 줄 건너뜀
                    if (lineIndex >= lines.Length) break;

                    string line = lines[lineIndex].Trim();
                    if (string.IsNullOrEmpty(line)) continue;

                    string[] values = line.Split(',');

                    if (values.Length < 3) continue;

                    float x = float.Parse(values[2]);
                    float y = float.Parse(values[3]);
                    float z = float.Parse(values[4]);
                    float visibility = float.Parse(values[5]);
                    Vector3 position = new Vector3(x * scale, y * scale, z * scale);
                    frame.Add(position);
                    frame_quality.Add(visibility);
                }

                frameDict[f] = frame;
            }

            // (2) frame, landmark 기반으로 다시 재구성
            for (int i = 1; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                if (string.IsNullOrEmpty(line)) continue;

                string[] values = line.Split(',');

                if (values.Length < 6)
                    continue; // 데이터 부족하면 스킵

                int frame = int.Parse(values[0]);
                int landmark = int.Parse(values[1]);
                float x = float.Parse(values[2]);
                float y = float.Parse(values[3]);
                float z = float.Parse(values[4]);
                float visibility = float.Parse(values[5]);

                Vector3 position = new Vector3(x * scale, y * scale, z * scale);

                // 프레임 단위로 리스트 초기화
                if (!frameDict.ContainsKey(frame))
                    frameDict[frame] = new List<Vector3>(new Vector3[33]); // BlazePose: 33개

                frameDict[frame][landmark] = position;
            }

            Debug.Log($"✅ Imported {frameDict.Count} frames of motion data");

            // 예시 출력: 첫 번째 프레임의 0번 랜드마크
            if (frameDict.ContainsKey(0))
            {
                Vector3 first = frameDict[0][0];
                Debug.Log($"Frame 0 -> Landmark 0 = {first}");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to import CSV: {e.Message}");
        }
    }
}