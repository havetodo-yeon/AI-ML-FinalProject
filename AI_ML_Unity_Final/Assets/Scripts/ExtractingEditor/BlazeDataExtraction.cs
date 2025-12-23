using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

public class BlazeDataExtraction : RealTimeAnimation
{
    public bool b_play = false;
    public int StartFrame = 1;
    public BlazePoseDataFile _BlazeMotionData;
    public Actor Character;

    public float human_size_scale = 5f;
    public float bone_size = 0.12f;

    public BlazePoseSkeletonBuilder skel_build_bp;

    // 회전 리타게팅용 캐시
    private Dictionary<int, Quaternion> boneRestRotations = new Dictionary<int, Quaternion>();
    private Dictionary<int, Vector3> boneRestDir = new Dictionary<int, Vector3>(); // 본이 바라보는 기본 방향
    private bool restPoseCaptured = false;

    // Runtime Smoothing
    [Range(0f, 1f)] public float pelvisPosSmooth = 0.25f;
    [Range(0f, 1f)] public float pelvisRotSmooth = 0.25f;

    private bool pelvisSmoothInit = false;
    private Vector3 pelvisPosSmoothed;
    private Quaternion pelvisRotSmoothed;

    // CSV(Blaze) 기준 첫 프레임 pelvis (Unity 변환 후)
    private bool basePelvisInitialized = false;
    private Vector3 basePelvisPosBP = Vector3.zero;
    private Quaternion basePelvisRotBP = Quaternion.identity;

    // 씬에서 캐릭터가 "재생 시작 시" 서 있던 pelvis(=Bones[2]) 월드 위치/회전
    private Vector3 basePelvisPosWorld = Vector3.zero;
    private Quaternion basePelvisRotWorld = Quaternion.identity;

    // Timeline World Offset
    public Transform worldOffset;   // 타임라인에서 움직일 오브젝트 (월드 기준)

    private bool offsetBaseInitialized = false;
    private Vector3 offsetBasePosWorld = Vector3.zero;
    private Quaternion offsetBaseRotWorld = Quaternion.identity;


    // Blaze index별로 "어느 자식을 보고 방향을 잡을지" 정의
    // (부모 Blaze → 자식 Blaze)
    private static readonly Dictionary<int, int> directionChildMap = new Dictionary<int, int>
    {
        {11, 13}, // LeftUpperArm  -> LeftElbow
        {13, 15}, // LeftLowerArm  -> LeftWrist
        {12, 14}, // RightUpperArm -> RightElbow
        {14, 16}, // RightLowerArm -> RightWrist
        {23, 25}, // LeftUpperLeg  -> LeftKnee
        {25, 27}, // LeftLowerLeg  -> LeftAnkle
        {24, 26}, // RightUpperLeg -> RightKnee
        {26, 28}, // RightLowerLeg -> RightAnkle
        {27, 31}, // LeftFoot -> LeftToes
        {28, 32}, // RightFoot -> RightToes
    };

    protected override void Setup()
    {
        if (_BlazeMotionData == null)
            _BlazeMotionData = ScriptableObject.CreateInstance<BlazePoseDataFile>();

        if (skel_build_bp == null)
            skel_build_bp = ScriptableObject.CreateInstance<BlazePoseSkeletonBuilder>();

        b_play = false;
    }

    protected override void Close()
    {

    }

    public void PrepareForPlayback()
    {
        if (_BlazeMotionData == null)
            _BlazeMotionData = ScriptableObject.CreateInstance<BlazePoseDataFile>();

        if (skel_build_bp == null)
            skel_build_bp = ScriptableObject.CreateInstance<BlazePoseSkeletonBuilder>();

        // CSV 자동 로드 보장
        _BlazeMotionData.EnsureLoaded();

        if (_BlazeMotionData.frameDict == null || _BlazeMotionData.frameDict.Count == 0)
        {
            Debug.LogError("[BlazeDataExtraction] Motion data not loaded. BlazePoseDataFile.defaultCsvFolderPath 확인.");
            return;
        }

        // 단일 캐릭터 방어
        if (Character == null)
        {
            Debug.LogError("[BlazeDataExtraction] Character(Source Actor)가 null 입니다.");
            return;
        }

        // 매핑 준비
        Animator anim = Character.GetComponentInChildren<Animator>();
        if (anim != null && anim.isHuman)
            skel_build_bp.BuildHumanoidMapping(Character, anim);
        else
            skel_build_bp.BuildMapping(Character);

        // 상태 초기화
        restPoseCaptured = false;
        Frame = StartFrame;

        // v2 anchor reset
        basePelvisInitialized = false;
        pelvisSmoothInit = false;

        offsetBaseInitialized = false;


    }

    public void PlayFromStart()
    {
        PrepareForPlayback();
        basePelvisInitialized = false;
        pelvisSmoothInit = false;
        b_play = true;
    }

    public void StopPlayback()
    {
        b_play = false;
    }

    //feed functions
    Vector3 ConvertBlazeToUnity(Vector3 bp)
    {
        // y축 반전, z축 방향 전환 (필요 시)
        return new Vector3(-bp.x, bp.y, -bp.z);
    }

    public Quaternion ComputeRotation(Vector3[] worldJointPos)
    {
        // 1. compute forward
        Vector3 leftShoulder = worldJointPos[11];
        Vector3 rightShoulder = worldJointPos[12];
        Vector3 leftHip = worldJointPos[23];
        Vector3 rightHip = worldJointPos[24];
        // 1.1. right vector
        Vector3 right_axis = rightShoulder - leftShoulder;
        right_axis[1] = 0; // horizontal projection of shoulder vector
        right_axis = right_axis.normalized;
        // 1.2. up vector
        Vector3 shoulder_center = (leftShoulder + rightShoulder)/2;
        Vector3 hip_center      = (leftHip + rightHip)/2;
        Vector3 up = (shoulder_center - hip_center).normalized;
        // 1.3. compute forward & orthogonalized up axis
        Vector3 forward_axis = Vector3.Cross(right_axis,up).normalized;
        Vector3 up_axis = Vector3.Cross(forward_axis,right_axis).normalized;

        // 2. compute rotation
        Matrix4x4 rotMatrix = Matrix4x4.identity;
        rotMatrix.SetColumn(0, new Vector4(right_axis.x, right_axis.y, right_axis.z, 0f));     // X axis
        rotMatrix.SetColumn(1, new Vector4(up_axis.x,    up_axis.y,    up_axis.z,    0f));     // Y axis
        rotMatrix.SetColumn(2, new Vector4(forward_axis.x, forward_axis.y, forward_axis.z, 0f)); // Z axis
        rotMatrix.SetColumn(3, new Vector4(0,0,0,1));
        
        return rotMatrix.rotation;
    }
    Vector3 ToPelvisLocal(Vector3 worldJoint, Quaternion pelvisRot, Vector3 pelvisPos)
    {
        // 1) translation 제거
        Vector3 p = worldJoint - pelvisPos;

        // 2) rotation 역변환 적용
        //   world → local 이므로 inverse 회전 적용
        return Quaternion.Inverse(pelvisRot) * p;
    }

    public void ComputeLocalPositions(Vector3[] worldJointPos, out Matrix4x4 origin, out Vector3[] localJointPos)
    {
        int hipL = 23;
        int hipR = 24;
        // pelvis position
        Vector3 hipPos = (worldJointPos[hipL] + worldJointPos[hipR]) * 0.5f;
        // pelvis rotation
        Quaternion hipRot = ComputeRotation(worldJointPos);
        // pelvis world origin
        Matrix4x4 pelvis = Matrix4x4.identity; 
        pelvis.SetTRS(hipPos,hipRot,pelvis.GetScale());
        origin = pelvis;
        // compute local positions
        Vector3[] _localJntPos = new Vector3[worldJointPos.Length];
        for (int i=0; i <worldJointPos.Length; i++)
        {
            _localJntPos[i] = ToPelvisLocal(worldJointPos[i],hipRot,hipPos);
        }
        localJointPos = _localJntPos;

    }
    public void UpdateBlazePose(BlazePoseSkeletonBuilder skel_bp, Vector3[] currentPose)
    {
        if (currentPose == null || currentPose.Length == 0) return;
        if (skel_bp == null || skel_bp.boneMap == null || skel_bp.boneMap.Count == 0) return;
        if (Character == null || Character.Bones == null) return;

        // 첫 프레임에 아직 캡처 안 했으면 T-pose 기준 정보 저장
        if (!restPoseCaptured)
        {
            CaptureRestPose(skel_bp);
        }

        // 1) BlazePose 좌표를 Unity 좌표로 모두 변환
        Vector3[] worldJointPos = new Vector3[currentPose.Length];
        for (int i = 0; i < currentPose.Length; i++)
        {
            worldJointPos[i] = ConvertBlazeToUnity(currentPose[i]) * human_size_scale;
        }

        // 2) compute pelvis origin
        Matrix4x4 pelvis = Matrix4x4.identity;
        Vector3[] local_positions = new Vector3[currentPose.Length];
        //ComputeLocalPositions(worldJointPos,out pelvis,out local_positions);
        //Character.Bones[2].Transform.SetPositionAndRotation(pelvis.GetPosition(),pelvis.rotation);
        ComputeLocalPositions(worldJointPos, out pelvis, out local_positions);

        Vector3 pPos = pelvis.GetPosition();
        Quaternion pRot = pelvis.rotation;

        // ===== "첫 프레임 pelvis" 기준으로 Δpos/Δrot 만들고
        //          "재생 시작 시 캐릭터 pelvis"에 더해서 월드 타겟을 만든다. =====
        if (!basePelvisInitialized)
        {
            basePelvisInitialized = true;

            // CSV 기준 첫 pelvis (Unity 변환된 좌표계)
            basePelvisPosBP = pPos;
            basePelvisRotBP = pRot;

            // 씬에서 캐릭터가 시작할 때 서 있던 pelvis(=Bones[2]) 월드 위치/회전
            basePelvisPosWorld = Character.Bones[2].Transform.position;
            basePelvisRotWorld = Character.Bones[2].Transform.rotation;
        }

        // CSV 기준 상대 이동/회전(첫 프레임 대비)
        Vector3 deltaPosBP = pPos - basePelvisPosBP;
        Quaternion deltaRotBP = Quaternion.Inverse(basePelvisRotBP) * pRot;

        // 최종 타겟 월드 pelvis = (씬 시작 pelvis) + Δ
        Vector3 targetPelvisPosWorld = basePelvisPosWorld + deltaPosBP;
        Quaternion targetPelvisRotWorld = basePelvisRotWorld * deltaRotBP;

        // Timeline World Offset (delta)
        Vector3 offsetDeltaPos = Vector3.zero;
        Quaternion offsetDeltaRot = Quaternion.identity;

        if (worldOffset != null)
        {
            if (!offsetBaseInitialized)
            {
                offsetBaseInitialized = true;
                offsetBasePosWorld = worldOffset.position;
                offsetBaseRotWorld = worldOffset.rotation;
            }

            offsetDeltaPos = worldOffset.position - offsetBasePosWorld;
            offsetDeltaRot = Quaternion.Inverse(offsetBaseRotWorld) * worldOffset.rotation;
        }

        // 최종 타겟(오프셋 포함)
        Vector3 finalTargetPos = targetPelvisPosWorld + offsetDeltaPos;
        Quaternion finalTargetRot = offsetDeltaRot * targetPelvisRotWorld;

        // smoothing (타겟 월드 pelvis를 부드럽게)
        if (!pelvisSmoothInit)
        {
            pelvisSmoothInit = true;
            pelvisPosSmoothed = finalTargetPos;
            pelvisRotSmoothed = finalTargetRot;
        }
        else
        {
            pelvisPosSmoothed = Vector3.Lerp(
                pelvisPosSmoothed,
                finalTargetPos,
                1f - Mathf.Pow(1f - pelvisPosSmooth, 60f * Time.deltaTime)
            );

            pelvisRotSmoothed = Quaternion.Slerp(
                pelvisRotSmoothed,
                finalTargetRot,
                1f - Mathf.Pow(1f - pelvisRotSmooth, 60f * Time.deltaTime)
            );
        }

        // 스무딩된 월드 pelvis 적용
        Character.Bones[2].Transform.SetPositionAndRotation(pelvisPosSmoothed, pelvisRotSmoothed);

        // 이후 bone 적용에서 쓰는 pelvis 회전도 "월드 pelvis"로 통일
        pelvis.SetTRS(pelvisPosSmoothed, pelvisRotSmoothed, pelvis.GetScale());

        // 3) 각 본의 회전을 방향 벡터 기반으로 갱신
        foreach (var kv in skel_bp.boneMap)
        {
            int parentBlaze = kv.Key;
            int parentBone = kv.Value;

            int childBlaze;
            if (!directionChildMap.TryGetValue(parentBlaze, out childBlaze))
                continue;

            if (parentBlaze < 0 || parentBlaze >= worldJointPos.Length) continue;
            if (childBlaze < 0 || childBlaze >= worldJointPos.Length) continue;

            if (!boneRestRotations.ContainsKey(parentBone) ||
                !boneRestDir.ContainsKey(parentBone))
                continue;

            // pelvis-local 기준 방향
            Vector3 targetDir = local_positions[childBlaze] - local_positions[parentBlaze];
            if (targetDir.sqrMagnitude < 1e-6f) continue;
            targetDir.Normalize();

            // restDir/restRot은 pelvis-local이어야 함 (아래 CaptureRestPose도 같이 바꿀 것)
            Vector3 restDir = boneRestDir[parentBone];
            Quaternion restRot = boneRestRotations[parentBone];

            // local에서 delta
            Quaternion delta = Quaternion.FromToRotation(restDir, targetDir);

            // 최종: "월드 pelvis" * (local bone rot)
            Quaternion finalWorldRot = pelvis.rotation * (delta * restRot);
            Character.Bones[parentBone].Transform.rotation = finalWorldRot;
        }
    }


    public void DrawBlazeSkel(BlazePoseSkeletonBuilder skel_bp, float boneSize, Color boneColor)
    {
        // 1. boneMap이 없거나 비어있으면 바로 리턴
        if (skel_bp == null || skel_bp.boneMap == null || skel_bp.boneMap.Count == 0)
        {
            Debug.LogWarning("[DrawBlazeSkel] boneMap is not initialized — skipping draw.");
            return;
        }

        // 2. blazePoseBones도 체크
        if (skel_bp.blazePoseBones == null || skel_bp.blazePoseBones.GetLength(0) == 0)
        {
            Debug.LogWarning("[DrawBlazeSkel] blazePoseBones is not defined — skipping draw.");
            return;
        }

        UltiDraw.Begin();

        for (int i = 0; i < skel_bp.blazePoseBones.GetLength(0); i++)
        {
            int parent = skel_bp.blazePoseBones[i, 0];
            int child = skel_bp.blazePoseBones[i, 1];

            int map_parent = skel_bp.boneMap[parent];
            int map_child = skel_bp.boneMap[child];

            if (parent < 0 || parent >= Character.Bones.Length) continue;
            if (child < 0 || child >= Character.Bones.Length) continue;
            if (Character.Bones[map_parent] == null || Character.Bones[map_child] == null) continue;

            Vector3 parentPos = Character.Bones[map_parent].Transform.position;
            Vector3 childPos = Character.Bones[map_child].Transform.position;

            float length = Vector3.Distance(parentPos, childPos);

            //UltiDraw.DrawLine(parentPos, childPos,boneSize, boneColor);

            UltiDraw.DrawBone(
                parentPos,
                Quaternion.FromToRotation(Vector3.forward, (childPos - parentPos).normalized),
                12.5f * boneSize * length,
                length,
                Color.grey
            ); // boneColor.Transparent(1f)
        }

        UltiDraw.End();
    }
    
    /// <summary>
    /// 캐릭터 T-pose 상태에서
    /// - 각 본의 기본 회전
    /// - 각 본이 바라보는 기본 방향(부모→자식)
    /// 을 한 번만 저장
    /// </summary>
    public void CaptureRestPose(BlazePoseSkeletonBuilder skel_bp)
    {
        boneRestRotations.Clear();
        boneRestDir.Clear();
        restPoseCaptured = false;

        // pelvis 기준 (Bones[2]를 pelvis로 사용중)
        Transform pelvisT = Character.Bones[2].Transform;
        Quaternion pelvisRot0 = pelvisT.rotation;
        Quaternion invPelvisRot0 = Quaternion.Inverse(pelvisRot0);

        if (Character == null || Character.Bones == null)
        {
            Debug.LogWarning("[BlazeDataExtraction] Character is null, cannot capture rest pose.");
            return;
        }

        if (skel_bp == null || skel_bp.boneMap == null || skel_bp.boneMap.Count == 0)
        {
            Debug.LogWarning("[BlazeDataExtraction] skel_bp / boneMap not ready.");
            return;
        }

        // 1) 각 본의 월드 기준 기본 회전 저장
        foreach (var kv in skel_bp.boneMap)
        {
            int boneIndex = kv.Value;
            if (boneIndex < 0 || boneIndex >= Character.Bones.Length) continue;

            Transform t = Character.Bones[boneIndex].Transform;
            boneRestRotations[boneIndex] = invPelvisRot0 * t.rotation;
        }

        // 2) 각 본이 바라보는 방향 저장
        foreach (var kv in skel_bp.boneMap)
        {
            int parentBlaze = kv.Key;
            int parentBone = kv.Value;

            int childBlaze;
            if (!directionChildMap.TryGetValue(parentBlaze, out childBlaze))
                continue; // Blaze joint는 방향을 정의X

            if (!skel_bp.boneMap.ContainsKey(childBlaze))
                continue; // 자식 Blaze가 매핑돼 있지 않으면 스킵

            int childBone = skel_bp.boneMap[childBlaze];

            Transform tParent = Character.Bones[parentBone].Transform;
            Transform tChild = Character.Bones[childBone].Transform;

            Vector3 dirWorld = tChild.position - tParent.position;
            if (dirWorld.sqrMagnitude < 1e-6f) continue;

            Vector3 dirLocal = invPelvisRot0 * dirWorld;
            dirLocal.Normalize();
            boneRestDir[parentBone] = dirLocal;
        }


        restPoseCaptured = true;
        Debug.Log("[BlazeDataExtraction] Rest pose captured.");
    }


    protected override void Feed()
    {
        // Vector3 vector = _BlazeMotionData.frameDict[0][0];
        // Debug.Log("see " + vector);
        if (b_play)
        {

            // import single data from csvList;
            if (Frame == StartFrame)
            {
                _BlazeMotionData.ImportCSVData(_BlazeMotionData.selectedData, 1.0f);
            }

            // initialize the data
            if (Frame >= _BlazeMotionData.frameDict.Count)
            {
                b_play = false;
                return;
            }
            else
            {
                Vector3[] currentPose = _BlazeMotionData.frameDict[Frame].ToArray();

                //Debug.Log($" Frame : {Frame} , currentPose {currentPose[0]} / {currentPose.Length}");
                UpdateBlazePose(skel_build_bp, currentPose);
                
                Frame++;
            }
        }

    }
    protected override void Read()
    {
    }
    protected override void Postprocess()
    { }
    protected override void OnGUIDerived()
    { }
    protected override void OnRenderObjectDerived()
    {
        //DrawBlazeSkel(skel_build_bp, bone_size, Color.cyan);
    }
    
    public void Prepare()
    {
        // 런타임에서 누락되는 것들 보정
        if (_BlazeMotionData == null)
            _BlazeMotionData = ScriptableObject.CreateInstance<BlazePoseDataFile>();

        if (skel_build_bp == null)
            skel_build_bp = ScriptableObject.CreateInstance<BlazePoseSkeletonBuilder>();

        if (Character == null)
        {
            Debug.LogError("[BlazeDataExtraction] Character(Source Actor)가 null 입니다.");
            return;
        }

        // Humanoid 매핑 준비
        Animator anim = Character.GetComponentInChildren<Animator>();
        if (anim != null && anim.isHuman)
            skel_build_bp.BuildHumanoidMapping(Character, anim);
        else
            skel_build_bp.BuildMapping(Character);

        // RestPose 다시 캡처하게
        // (private라면 bool만 초기화하는 방식으로 둬도 됨)
        // restPoseCaptured = false; // <- 현재 필드가 private 이지만 같은 클래스 내부니까 OK
        restPoseCaptured = false;

        // 시작 프레임으로 리셋
        Frame = StartFrame;
    }

    [CustomEditor(typeof(BlazeDataExtraction), true)]
    public class BlazeDataExtraction_Editor : Editor
    {
        public BlazeDataExtraction Target;
        public void Awake()
        {
            Target = (BlazeDataExtraction)target;
            //Target.is_random = true;
            Target.skel_build_bp = ScriptableObject.CreateInstance<BlazePoseSkeletonBuilder>();
        }
        public override void OnInspectorGUI()
        {
            Inspector();
        }
        private void Inspector()
        {
            Utility.ResetGUIColor();
            Utility.SetGUIColor(UltiDraw.LightGrey);

            // Assigning Target Avatar
            EditorGUILayout.BeginVertical();
            Target.Character = (Actor)EditorGUILayout.ObjectField("Source Actor", Target.Character, typeof(Actor), true);
            EditorGUILayout.EndVertical();

            if (Target._BlazeMotionData != null)
            {
                //BlazeData
                Target._BlazeMotionData.MotionCSVFile_Inspector(Target.Character);
            }

            Target.human_size_scale = EditorGUILayout.FloatField("human_scale", Target.human_size_scale);
            
            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField("Timeline World Offset (optional)", EditorStyles.boldLabel);
            Target.worldOffset = (Transform)EditorGUILayout.ObjectField(
                "World Offset Transform",
                Target.worldOffset,
                typeof(Transform),
                true
            );
            //if (GUI.changed)
            //{
            //    EditorUtility.SetDirty(Target);
            //}

            EditorGUILayout.Space(10);

            // play button
            if (Utility.GUIButton("reset & play animation", Color.white, Color.red))
            {
                if (Target.Character != null)
                {
                    Animator anim = Target.Character.GetComponentInChildren<Animator>();

                    if (anim != null && anim.isHuman)
                    {
                        // 휴머노이드 캐릭터라면 자동 Humanoid 매핑 사용
                        Target.skel_build_bp.BuildHumanoidMapping(Target.Character, anim);
                    }
                    else
                    {
                        // 아니면 기존 매핑 사용
                        Target.skel_build_bp.BuildMapping(Target.Character);
                    }
                }

                Target.Frame = Target.StartFrame;
                Target.b_play = true;
            }

        }

    }
}