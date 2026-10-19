import cv2

print("=" * 60)
print("CAMERA BACKEND TEST")
print("=" * 60)

for camera_id in range(5):

    print(f"\n[TEST] Camera {camera_id}")

    camera = cv2.VideoCapture(
        camera_id,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        print("[NO] Could not open")
        camera.release()
        continue

    print("[OK] Opened using DirectShow")

    success, frame = camera.read()

    if success:
        print(
            f"[OK] Frame received: "
            f"{frame.shape[1]}x{frame.shape[0]}"
        )

        cv2.imshow(
            f"Camera {camera_id}",
            frame
        )

        print("Press any key to continue...")
        cv2.waitKey(2000)

        cv2.destroyAllWindows()

    else:
        print("[ERROR] Camera opened but frame failed")

    camera.release()

cv2.destroyAllWindows()

print("\n[INFO] Camera test completed.")